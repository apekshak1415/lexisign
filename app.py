from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import pickle
import threading
import time
import os
from collections import deque, Counter

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

# ── Load models ───────────────────────────────────────────────────────────────
letter_model = word_model = le_letters = le_words = None

def load_models():
    global letter_model, word_model, le_letters, le_words
    try:
        from tensorflow.keras.models import load_model
        letter_model = load_model(os.path.join(BASE, 'models/letter_model.h5'))
        word_model   = load_model(os.path.join(BASE, 'models/word_model.h5'))
        with open(os.path.join(BASE, 'models/label_encoder_letters.pkl'), 'rb') as f:
            le_letters = pickle.load(f)
        with open(os.path.join(BASE, 'models/label_encoder_words.pkl'), 'rb') as f:
            le_words = pickle.load(f)
        print(f"✅ {len(le_letters.classes_)} letters, {len(le_words.classes_)} words loaded")
    except Exception as e:
        print(f"[WARN] Model load: {e}")

load_models()

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

hands_sol    = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
holistic_sol = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
POSE_UPPER   = [11,12,13,14,15,16,23,24,25,26,27,28]

LANGUAGES = ["English","Hindi","Telugu","Tamil","Bengali",
             "Kannada","Malayalam","Marathi","Punjabi","Urdu","Kashmiri"]

DICTIONARY = sorted(set([
    "HELLO","HELP","HAPPY","HUNGRY","THANK","TIRED","SORRY","STOP",
    "PLEASE","FOOD","FINE","WATER","WANT","GOOD","GO","NAME","NICE",
    "YES","YOU","NO","UNDERSTAND","COME","BAD","LIKE","LOVE","KNOW",
    "INDIA","ABOUT","AFTER","ALL","AND","ARE","AT","BE","BECAUSE",
    "BIG","BUT","CAN","DAY","DO","DOWN","EACH","EVEN","EVERY","FEEL",
    "FOR","GET","GIVE","HAS","HAVE","HERE","IF","IN","IS","IT","JUST",
    "KNOW","LAST","LONG","LOOK","MAKE","MANY","ME","MORE","MY","NEED",
    "NEW","NOW","OF","ON","ONE","ONLY","OR","OUR","OUT","PEOPLE",
    "RIGHT","SAY","SEE","SHE","SO","SOME","STILL","TAKE","THAN",
    "THEIR","THEM","THEN","THERE","THEY","THIS","TIME","TO","TOO",
    "TWO","UP","US","USE","VERY","WAY","WE","WELL","WHEN","WHICH",
    "WHO","WILL","WITH","WOULD",
]))

# ── Word groups (Point 4: real daily-life phrases) ────────────────────────────
WORD_GROUPS = {
    "Greetings": [
        "HELLO", "GOODBYE", "GOOD MORNING", "GOOD AFTERNOON",
        "GOOD EVENING", "WELCOME", "THANK YOU", "SORRY", "PLEASE", "YES", "NO"
    ],
    "People & Pronouns": [
        "I", "YOU", "HE", "SHE", "WE", "THEY",
        "MY NAME", "FRIEND", "FAMILY", "MOTHER", "FATHER"
    ],
    "Questions": [
        "WHAT", "WHERE", "WHEN", "WHO", "WHY", "HOW",
        "PLACE", "THIS", "WHICH", "DO YOU"
    ],
    "Common Verbs": [
        "COME", "GO", "EAT", "DRINK", "SLEEP", "HELP",
        "WANT", "NEED", "KNOW", "UNDERSTAND", "STOP", "WAIT"
    ],
    "Feelings & States": [
        "HAPPY", "SAD", "ANGRY", "TIRED", "SICK",
        "HUNGRY", "THIRSTY", "PAIN", "FINE", "SCARED"
    ],
    "Places & Things": [
        "HOME", "HOSPITAL", "SCHOOL", "BATHROOM", "FOOD",
        "WATER", "MEDICINE", "MONEY", "TIME", "PHONE"
    ],
    "Descriptors": [
        "GOOD", "BAD", "BIG", "SMALL", "FAST",
        "SLOW", "HOT", "COLD", "MORE", "ENOUGH"
    ],
}

# ── Translator state ──────────────────────────────────────────────────────────
state = {
    "mode": "LETTER", "language": "English",
    "word_buffer": [], "sentence_words": [],
    "final_sentence": "", "translated_sentence": "",
    "detected": "", "confidence": 0.0,
    "suggestions": [], "word_suggestions": [],
    "camera_on": False, "hold_progress": 0,
    "recent_signs": [], "nlp_loading": False,
}

frame_sequence    = deque(maxlen=30)
word_pred_history = deque(maxlen=5)
prev_word_feats   = None
word_frame_count  = 0
last_word = last_letter = ""
letter_hold_count = 0

# ── Collect state ─────────────────────────────────────────────────────────────
collect_state = {
    "active": False,
    "word": "",
    "clip_count": 0,
    "target_clips": 30,
    "status": "idle",   # idle | countdown | recording | done
    "countdown": 0,
    "sequences": [],
    "labels": [],
}

cap            = None
collect_cap    = None
output_frame   = None
collect_frame  = None
frame_lock     = threading.Lock()
collect_lock   = threading.Lock()

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_letter_features(hr):
    features = []
    detected = hr.multi_hand_landmarks or []
    for i in range(2):
        if i < len(detected):
            hl = detected[i]
            xs = [lm.x for lm in hl.landmark]
            ys = [lm.y for lm in hl.landmark]
            mx, my = min(xs), min(ys)
            for lm in hl.landmark:
                features.extend([lm.x-mx, lm.y-my])
        else:
            features.extend([0.0]*42)
    return features

def extract_holistic_features(results):
    def hand_lm(hand):
        if hand:
            pts = np.array([[lm.x,lm.y,lm.z] for lm in hand.landmark]).flatten()
            pts[0::3]-=pts[0]; pts[1::3]-=pts[1]
            return pts
        return np.zeros(63)
    def pose_lm(pose):
        if pose:
            pts = np.array([[pose.landmark[i].x,pose.landmark[i].y,pose.landmark[i].z]
                            for i in POSE_UPPER]).flatten()
            sx=(pose.landmark[11].x+pose.landmark[12].x)/2
            sy=(pose.landmark[11].y+pose.landmark[12].y)/2
            pts[0::3]-=sx; pts[1::3]-=sy
            return pts
        return np.zeros(36)
    return np.concatenate([hand_lm(results.right_hand_landmarks),
                           hand_lm(results.left_hand_landmarks),
                           pose_lm(results.pose_landmarks)])

def hand_visible(r):
    return r.right_hand_landmarks is not None or r.left_hand_landmarks is not None

def add_velocity(sequence):
    deltas = np.zeros_like(sequence)
    deltas[1:] = sequence[1:] - sequence[:-1]
    return np.concatenate([sequence, deltas], axis=1)

def get_suggestions(prefix, n=4):
    if not prefix: return []
    return [w for w in DICTIONARY if w.startswith(prefix.upper())][:n]

def augment_sequence(seq, n=7):
    augmented = []
    for _ in range(n):
        aug = seq.copy()
        aug += np.random.normal(0, 0.005, aug.shape)
        aug *= np.random.uniform(0.9, 1.1)
        if np.random.rand() > 0.5:
            idx = np.random.randint(1, len(aug)-1)
            aug = np.delete(aug, idx, axis=0)
            aug = np.insert(aug, idx, aug[idx], axis=0)
        augmented.append(aug)
    return augmented

# ── Translator camera loop ────────────────────────────────────────────────────
def camera_loop():
    global cap, output_frame, prev_word_feats, word_frame_count
    global letter_hold_count, last_letter, last_word

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    while state["camera_on"]:
        ret, frame = cap.read()
        if not ret: break
        frame   = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if state["mode"] == "LETTER":
            hr = hands_sol.process(img_rgb)
            if hr.multi_hand_landmarks:
                for hl in hr.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,120), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255,220,0), thickness=2))
                feats = np.array(extract_letter_features(hr)).reshape(1,-1)
                pred  = letter_model.predict(feats, verbose=0)
                idx   = np.argmax(pred); conf = float(pred[0][idx])
                label = le_letters.classes_[idx]
                if conf >= 0.75:
                    state["detected"] = label; state["confidence"] = conf
                    if label == last_letter: letter_hold_count += 1
                    else: letter_hold_count = 0; last_letter = label
                    state["hold_progress"] = letter_hold_count / 10
                    if letter_hold_count >= 10:
                        state["word_buffer"].append(label)
                        state["suggestions"] = get_suggestions("".join(state["word_buffer"]))
                        letter_hold_count = 0; last_letter = ""
                else:
                    state["detected"] = ""; state["hold_progress"] = 0
            else:
                letter_hold_count = 0; state["hold_progress"] = 0
        else:
            hol = holistic_sol.process(img_rgb)
            if hol.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hol.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,120), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255,220,0), thickness=2))
            if hol.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hol.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255,100,0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255,200,0), thickness=2))
            if hand_visible(hol):
                ff  = extract_holistic_features(hol)
                vel = ff - prev_word_feats if prev_word_feats is not None else np.zeros(162)
                prev_word_feats = ff.copy()
                frame_sequence.append(np.concatenate([ff, vel]))
                word_frame_count += 1
                if len(frame_sequence) == 30 and word_frame_count % 5 == 0:
                    arr  = np.array(list(frame_sequence), dtype=np.float32).reshape(1,30,324)
                    pred = word_model.predict(arr, verbose=0)[0]

                    # Get top 3 predictions above 80% confidence
                    top_indices = np.argsort(pred)[::-1][:3]
                    top_preds   = [(le_words.classes_[i], float(pred[i]))
                                   for i in top_indices if pred[i] >= 0.80]

                    if top_preds:
                        best_label, best_conf = top_preds[0]
                        state["detected"]   = best_label
                        state["confidence"] = best_conf

                        # Show top predictions as word suggestions
                        state["word_suggestions"] = [
                            {"word": w, "conf": round(c*100)} for w, c in top_preds
                        ]
                    else:
                        state["word_suggestions"] = []
                        state["detected"] = ""
            else:
                frame_sequence.clear(); prev_word_feats = None
                word_pred_history.clear(); word_frame_count = 0

        with frame_lock:
            output_frame = frame.copy()

    if cap: cap.release()

def generate_frames():
    blank = np.zeros((540,960,3), dtype=np.uint8)
    while True:
        with frame_lock:
            frame = output_frame.copy() if output_frame is not None else blank.copy()
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.033)

# ── Collect camera loop ───────────────────────────────────────────────────────
FRAMES_PER_CLIP = 30

def collect_camera_loop():
    global collect_cap, collect_frame

    collect_cap = cv2.VideoCapture(0)
    collect_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    collect_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as hol:
        while collect_state["active"]:
            ret, frame = collect_cap.read()
            if not ret: break
            frame   = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hol.process(img_rgb)

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,120), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255,220,0), thickness=2))
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,100,0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255,200,0), thickness=2))

            # Overlay status
            word   = collect_state["word"]
            clip   = collect_state["clip_count"]
            target = collect_state["target_clips"]
            status = collect_state["status"]

            # Top bar
            cv2.rectangle(frame, (0,0), (960,60), (15,15,30), -1)
            cv2.putText(frame, f"Word: {word}", (20,38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.putText(frame, f"Clip {clip}/{target}", (700,38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,180,255), 2)

            if status == "countdown":
                cd = collect_state["countdown"]
                cv2.putText(frame, str(cd), (450,300),
                            cv2.FONT_HERSHEY_SIMPLEX, 6, (0,220,255), 8)
                cv2.putText(frame, "Get Ready!", (340,380),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,220,255), 3)

            elif status == "recording":
                # Recording progress bar
                prog = len(collect_state.get("current_clip_frames", [])) / FRAMES_PER_CLIP
                bw   = int(900 * prog)
                cv2.rectangle(frame, (20,480), (940,500), (40,40,40), -1)
                cv2.rectangle(frame, (20,480), (20+bw,500), (0,220,100), -1)
                cv2.putText(frame, "RECORDING...", (20,470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,100), 2)

            elif status == "idle":
                cv2.putText(frame, "Press RECORD to start clip", (250,300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150,150,180), 2)

            elif status == "done":
                cv2.putText(frame, f"✓ {target} clips collected! Click Save.", (150,300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,120), 2)

            # Recording: capture frames
            if status == "recording":
                if "current_clip_frames" not in collect_state:
                    collect_state["current_clip_frames"] = []
                def hand_lm(hand):
                    if hand:
                        pts = np.array([[lm.x,lm.y,lm.z] for lm in hand.landmark]).flatten()
                        pts[0::3]-=pts[0]; pts[1::3]-=pts[1]
                        return pts
                    return np.zeros(63)
                def pose_lm(pose):
                    if pose:
                        pts = np.array([[pose.landmark[i].x,pose.landmark[i].y,pose.landmark[i].z]
                                        for i in POSE_UPPER]).flatten()
                        sx=(pose.landmark[11].x+pose.landmark[12].x)/2
                        sy=(pose.landmark[11].y+pose.landmark[12].y)/2
                        pts[0::3]-=sx; pts[1::3]-=sy
                        return pts
                    return np.zeros(36)

                feats = np.concatenate([hand_lm(results.right_hand_landmarks),
                                        hand_lm(results.left_hand_landmarks),
                                        pose_lm(results.pose_landmarks)])
                collect_state["current_clip_frames"].append(feats)

                if len(collect_state["current_clip_frames"]) >= FRAMES_PER_CLIP:
                    seq  = np.array(collect_state["current_clip_frames"])
                    seq  = add_velocity(seq)
                    collect_state["sequences"].append(seq)
                    collect_state["labels"].append(word)
                    for aug in augment_sequence(seq, 7):
                        collect_state["sequences"].append(aug)
                        collect_state["labels"].append(word)
                    collect_state["clip_count"] += 1
                    collect_state["current_clip_frames"] = []
                    if collect_state["clip_count"] >= target:
                        collect_state["status"] = "done"
                    else:
                        collect_state["status"] = "idle"

            with collect_lock:
                collect_frame = frame.copy()

    if collect_cap: collect_cap.release()

def generate_collect_frames():
    blank = np.zeros((540,960,3), dtype=np.uint8)
    while True:
        with collect_lock:
            frame = collect_frame.copy() if collect_frame is not None else blank.copy()
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.033)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
                           languages=LANGUAGES,
                           letter_classes=len(le_letters.classes_) if le_letters else 0,
                           word_classes=len(le_words.classes_)   if le_words   else 0)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/collect_feed')
def collect_feed():
    return Response(generate_collect_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera', methods=['POST'])
def toggle_camera():
    action = request.json.get('action')
    if action == 'start' and not state["camera_on"]:
        state["camera_on"] = True
        threading.Thread(target=camera_loop, daemon=True).start()
    elif action == 'stop':
        state["camera_on"] = False
    return jsonify({"camera_on": state["camera_on"]})

@app.route('/api/state')
def get_state():
    return jsonify({
        "detected": state["detected"],
        "confidence": round(state["confidence"]*100),
        "mode": state["mode"], "language": state["language"],
        "word_buffer": "".join(state["word_buffer"]),
        "sentence": " › ".join(state["sentence_words"]),
        "final": state["final_sentence"],
        "translated": state["translated_sentence"],
        "suggestions": state["suggestions"],
        "word_suggestions": state["word_suggestions"],
        "hold": round(state["hold_progress"]*100),
        "recent": state["recent_signs"],
        "camera_on": state["camera_on"],
        "nlp_loading": state["nlp_loading"],
    })

@app.route('/api/action', methods=['POST'])
def action():
    global last_word, last_letter, letter_hold_count
    cmd = request.json.get('cmd')

    if cmd == 'mode':
        state["mode"] = "WORD" if state["mode"] == "LETTER" else "LETTER"
        frame_sequence.clear(); word_pred_history.clear()
        letter_hold_count = 0; last_letter = last_word = ""
        state["detected"] = ""; state["hold_progress"] = 0
        state["word_suggestions"] = []

    elif cmd == 'clear':
        state["word_buffer"].clear(); state["sentence_words"].clear()
        state["final_sentence"] = ""; state["translated_sentence"] = ""
        state["suggestions"] = []; state["word_suggestions"] = []
        state["detected"] = ""
        last_word = last_letter = ""; letter_hold_count = 0
        frame_sequence.clear(); word_pred_history.clear()

    elif cmd == 'confirm_word_suggestion':
        # User picked one of the word suggestions
        word = request.json.get('word', '')
        if word:
            state["sentence_words"].append(word)
            state["recent_signs"].insert(0, word)
            state["recent_signs"] = state["recent_signs"][:5]
            state["word_suggestions"] = []
            state["detected"] = ""
            frame_sequence.clear()
            word_pred_history.clear()

    elif cmd == 'backspace':
        if state["word_buffer"]:
            state["word_buffer"].pop()
            state["suggestions"] = get_suggestions("".join(state["word_buffer"]))
        elif state["sentence_words"]:
            state["sentence_words"].pop()

    elif cmd == 'confirm_word':
        if state["word_buffer"]:
            word = "".join(state["word_buffer"])
            state["sentence_words"].append(word)
            state["recent_signs"].insert(0, word)
            state["recent_signs"] = state["recent_signs"][:5]
            state["word_buffer"].clear(); state["suggestions"] = []

    elif cmd == 'autocomplete':
        idx = request.json.get('idx', 0)
        if idx < len(state["suggestions"]):
            chosen = state["suggestions"][idx]
            state["sentence_words"].append(chosen)
            state["recent_signs"].insert(0, chosen)
            state["recent_signs"] = state["recent_signs"][:5]
            state["word_buffer"].clear(); state["suggestions"] = []

    elif cmd == 'generate':
        # Guard: don't run if already loading or already has result for same words
        if state["nlp_loading"]:
            return jsonify({"ok": True, "skipped": "already loading"})
        words_snapshot = list(state["sentence_words"])
        if not words_snapshot:
            return jsonify({"ok": True, "skipped": "no words"})
        def _run():
            state["nlp_loading"] = True
            state["final_sentence"]      = ""
            state["translated_sentence"] = ""
            try:
                import sys
                sys.path.insert(0, os.path.join(BASE, 'src'))
                from nlp_helper import correct_sentence
                from tts_helper  import speak, translate_text
                # Pass snapshot so even if sentence_words changes, result is consistent
                corrected = correct_sentence(words_snapshot)
                state["final_sentence"]      = corrected
                state["translated_sentence"] = translate_text(corrected, state["language"])
                threading.Thread(target=speak,
                                 args=(corrected, state["language"]),
                                 daemon=True).start()
            except Exception as e:
                state["final_sentence"] = " ".join(words_snapshot) + "."
                print(f"NLP error: {e}")
            state["nlp_loading"] = False
        threading.Thread(target=_run, daemon=True).start()

    elif cmd == 'language':
        state["language"] = request.json.get('lang', 'English')

    return jsonify({"ok": True})

# ── Collect routes ────────────────────────────────────────────────────────────
@app.route('/api/collect/start', methods=['POST'])
def collect_start():
    word = request.json.get('word', '')
    if not word: return jsonify({"error": "No word"}), 400

    # Load existing data to check count
    data_path = os.path.join(BASE, 'data/own_word_landmarks.pkl')
    existing  = 0
    if os.path.exists(data_path):
        with open(data_path,'rb') as f:
            d = pickle.load(f)
        existing = list(d['labels']).count(word)

    collect_state.update({
        "active": True, "word": word,
        "clip_count": existing // 8,   # existing raw clips (÷8 augmentation)
        "target_clips": 30,
        "status": "idle",
        "sequences": [], "labels": [],
    })
    collect_state.pop("current_clip_frames", None)

    if not any(t.name == "collect_loop" and t.is_alive()
               for t in threading.enumerate()):
        t = threading.Thread(target=collect_camera_loop, daemon=True, name="collect_loop")
        t.start()

    return jsonify({"ok": True, "existing": existing // 8})

@app.route('/api/collect/record', methods=['POST'])
def collect_record():
    """Start recording one clip with 3-second countdown."""
    if not collect_state["active"]:
        return jsonify({"error": "Not active"}), 400
    if collect_state["status"] in ("recording","countdown","done"):
        return jsonify({"error": "Already recording"}), 400

    def _countdown():
        for i in range(3, 0, -1):
            collect_state["status"]    = "countdown"
            collect_state["countdown"] = i
            time.sleep(0.8)
        collect_state["status"] = "recording"
        collect_state.pop("current_clip_frames", None)

    threading.Thread(target=_countdown, daemon=True).start()
    return jsonify({"ok": True})

@app.route('/api/collect/save', methods=['POST'])
def collect_save():
    """Save collected clips to pkl file."""
    if not collect_state["sequences"]:
        return jsonify({"error": "No clips to save"}), 400

    data_path = os.path.join(BASE, 'data/own_word_landmarks.pkl')
    os.makedirs(os.path.join(BASE, 'data'), exist_ok=True)

    all_seqs = list(collect_state["sequences"])
    all_lbls = list(collect_state["labels"])

    if os.path.exists(data_path):
        with open(data_path,'rb') as f:
            existing = pickle.load(f)
        all_seqs = list(existing['sequences']) + all_seqs
        all_lbls = list(existing['labels'])    + all_lbls

    with open(data_path,'wb') as f:
        pickle.dump({'sequences': np.array(all_seqs),
                     'labels':    np.array(all_lbls)}, f)

    word  = collect_state["word"]
    count = collect_state["clip_count"]
    collect_state.update({"active":False,"status":"idle","sequences":[],"labels":[]})

    return jsonify({"ok": True, "word": word, "clips": count})

@app.route('/api/collect/stop', methods=['POST'])
def collect_stop():
    collect_state["active"] = False
    return jsonify({"ok": True})

@app.route('/api/collect/status')
def collect_status():
    return jsonify({
        "status":    collect_state["status"],
        "clip_count": collect_state["clip_count"],
        "target":    collect_state["target_clips"],
        "word":      collect_state["word"],
        "countdown": collect_state.get("countdown", 0),
    })

@app.route('/api/collect/words')
def get_collect_words():
    try:
        data_path = os.path.join(BASE, 'data/own_word_landmarks.pkl')
        counts = Counter()
        if os.path.exists(data_path):
            with open(data_path,'rb') as f:
                d = pickle.load(f)
            counts = Counter(d['labels'])
    except: counts = Counter()

    result = []
    for group, words in WORD_GROUPS.items():
        result.append({"group": group, "words": [
            {"word": w, "count": counts.get(w, 0)} for w in words
        ]})
    return jsonify(result)

@app.route('/api/model/stats')
def model_stats():
    return jsonify({
        "letter_classes": len(le_letters.classes_) if le_letters else 0,
        "word_classes":   len(le_words.classes_)   if le_words   else 0,
        "letter_words":   list(le_letters.classes_) if le_letters else [],
        "word_words":     list(le_words.classes_)   if le_words   else [],
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)