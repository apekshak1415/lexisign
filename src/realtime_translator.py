import cv2
import mediapipe as mp
import numpy as np
import pickle
import threading
from collections import deque
from tensorflow.keras.models import load_model
from nlp_helper import correct_sentence
from tts_helper import speak, setup_voice

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
print("📂 Loading models...")
letter_model = load_model('./models/letter_model.h5')
word_model   = load_model('./models/word_model.h5')

with open('./models/label_encoder_letters.pkl', 'rb') as f:
    le_letters = pickle.load(f)
with open('./models/label_encoder_words.pkl', 'rb') as f:
    le_words = pickle.load(f)

print(f"✅ Letter model: {len(le_letters.classes_)} classes")
print(f"✅ Word model:   {len(le_words.classes_)} classes")

# ─────────────────────────────────────────────
# DICTIONARY FOR AUTOCOMPLETE
# ─────────────────────────────────────────────
DICTIONARY = [
    # Common ISL words
    "HELLO","HELP","HAPPY","HUNGRY","HURT","HEAR","HEART","HIDING","HOW","HAD","HAPPENED",
    "THANK","TIRED","THINK","TALK","TELL","TODAY","TRUST","TRUTH","TRAIN","THIRSTY","THINGS","THAT",
    "SORRY","STOP","SLEEP","SITTING","SPEAK","SURE","SERVE","SHIRT","SLOWER","SOFTLY","STUBBORN",
    "PLEASE","PHONE","PLACE","POUR","PREPARE","PROMISE",
    "FOOD","FINE","FREE","FRIEND","FROM","FEVER","FAVOUR",
    "WATER","WANT","WEAR","WELCOME","WHAT","WHERE","WHO","WORRY",
    "GOOD","GO","GRATEFUL",
    "NAME","NICE","NOT","NUMBER",
    "YES","YOU",
    "NO","NICE",
    "UNDERSTAND",
    "REPEAT","ROOM","REALLY",
    "COME","COLD","COMB","CONGRATULATIONS","CRYING",
    "BAD","BEAUTIFUL","BECOME","BED","BORED","BRING",
    "MEDICINE","MEET",
    "KIND",
    "LEAVE","LIKE","LOVE",
    "OLD","OUTSIDE",
    "DO","DARE","DIFFERENCE","DILEMMA","DISAPPOINTED",
    "ENJOY","ENJOY",
    "AGREE","ALL","ANGRY","ANYTHING","APPRECIATE","AFRAID","ABUSE",
    "INDIA","INDIAN","ISL",
    # Common English words
    "ABOUT","AFTER","AGAIN","ALSO","ALWAYS","AND","ARE","AT",
    "BACK","BE","BECAUSE","BEEN","BEFORE","BIG","BUT","BY",
    "CAN","COME","COULD","DAY","DID","DOWN","EACH",
    "EVEN","EVERY","FEEL","FEW","FOR","FROM","GET","GIVE",
    "GREAT","HAS","HAVE","HER","HERE","HIM","HIS","HIS",
    "IF","IN","INTO","IS","IT","ITS","JUST","KNOW",
    "LAST","LITTLE","LONG","LOOK","MAKE","MANY","ME","MORE",
    "MOST","MY","NEED","NEW","NEXT","NOW","OF","OFF","ON",
    "ONE","ONLY","OR","OTHER","OUR","OUT","OVER","OWN",
    "PEOPLE","RIGHT","SAID","SAME","SAY","SEE","SHE","SHOULD",
    "SO","SOME","STILL","SUCH","TAKE","THAN","THEIR","THEM",
    "THEN","THERE","THEY","THIS","TIME","TO","TOO","TWO",
    "UP","US","USE","VERY","WAY","WE","WELL","WERE",
    "WHEN","WHICH","WHILE","WHO","WILL","WITH","WOULD",
]
DICTIONARY = sorted(set(DICTIONARY))

def get_autocomplete(prefix, max_suggestions=4):
    if not prefix:
        return []
    prefix = prefix.upper()
    return [w for w in DICTIONARY if w.startswith(prefix)][:max_suggestions]

# ─────────────────────────────────────────────
# MEDIAPIPE
# ─────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

setup_voice(rate=150)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
SEQUENCE_LENGTH      = 30     # updated: new word model uses 30 frames
LETTER_CONF_THRESH   = 0.75
WORD_CONF_THRESH     = 0.85   # raised from 0.70 → fewer false triggers
LETTER_STABLE_FRAMES = 10
WORD_STABLE_FRAMES   = 5      # prediction must be consistent this many times

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
mode              = "LETTER"
letter_hold_count = 0
last_letter       = ""
word_buffer       = []
sentence_words    = []
frame_sequence    = deque(maxlen=SEQUENCE_LENGTH)
word_frame_count  = 0
last_word         = ""
final_sentence    = ""
show_sentence     = False
suggestions       = []
selected_suggest  = -1

# Word mode: new stability + velocity state
word_pred_history  = deque(maxlen=WORD_STABLE_FRAMES)
prev_word_features = None

# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

# ── Letter landmarks — UNCHANGED from original ────────────────────────────────
def extract_letter_landmarks(hand_results):
    features = []
    detected = hand_results.multi_hand_landmarks or []
    for i in range(2):
        if i < len(detected):
            hl = detected[i]
            xs = [lm.x for lm in hl.landmark]
            ys = [lm.y for lm in hl.landmark]
            mx, my = min(xs), min(ys)
            for lm in hl.landmark:
                features.extend([lm.x - mx, lm.y - my])
        else:
            features.extend([0.0] * 42)
    return features

# ── Word landmarks — updated to match new model (162 features + velocity) ─────
POSE_UPPER = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

def extract_holistic_features(results):
    """
    162 features matching collect_own_word_data.py:
      right hand : 21 x 3 = 63  (wrist-normalized)
      left  hand : 21 x 3 = 63  (wrist-normalized)
      upper pose : 12 x 3 = 36  (shoulder-midpoint-normalized)
    """
    def hand_lm(hand):
        if hand:
            pts = np.array([[lm.x, lm.y, lm.z]
                            for lm in hand.landmark]).flatten()
            pts[0::3] -= pts[0]
            pts[1::3] -= pts[1]
            return pts
        return np.zeros(63)

    def pose_lm(pose):
        if pose:
            pts = np.array([[pose.landmark[i].x,
                             pose.landmark[i].y,
                             pose.landmark[i].z]
                            for i in POSE_UPPER]).flatten()
            sx = (pose.landmark[11].x + pose.landmark[12].x) / 2
            sy = (pose.landmark[11].y + pose.landmark[12].y) / 2
            pts[0::3] -= sx
            pts[1::3] -= sy
            return pts
        return np.zeros(36)

    rh = hand_lm(results.right_hand_landmarks)
    lh = hand_lm(results.left_hand_landmarks)
    ps = pose_lm(results.pose_landmarks)
    return np.concatenate([rh, lh, ps])   # (162,)

def hand_visible(hol_results):
    """True only when at least one hand is detected."""
    return (hol_results.right_hand_landmarks is not None or
            hol_results.left_hand_landmarks  is not None)

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict_letter(hand_results):
    feats = np.array(extract_letter_landmarks(hand_results)).reshape(1, -1)
    pred  = letter_model.predict(feats, verbose=0)
    idx   = np.argmax(pred)
    conf  = float(pred[0][idx])
    return le_letters.classes_[idx], conf

def predict_word(sequence):
    arr  = np.array(sequence, dtype=np.float32).reshape(1, SEQUENCE_LENGTH, 324)
    pred = word_model.predict(arr, verbose=0)
    idx  = np.argmax(pred)
    conf = float(pred[0][idx])
    return le_words.classes_[idx], conf

# ─────────────────────────────────────────────
# UI — UNCHANGED from original
# ─────────────────────────────────────────────
def draw_ui(frame, detected, confidence, mode, word_buffer,
            sentence_words, final_sentence, show_sentence,
            suggestions, selected_suggest, hold_progress):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    mode_color = (34, 177, 76) if mode == "LETTER" else (0, 120, 255)
    cv2.rectangle(frame, (10, 10), (145, 60), mode_color, -1)
    cv2.putText(frame, f"MODE: {mode}", (18, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.putText(frame, f"{detected}  {confidence:.0%}", (160, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)

    if mode == "LETTER" and hold_progress > 0:
        bar_w = int((hold_progress / LETTER_STABLE_FRAMES) * 180)
        cv2.rectangle(frame, (160, 40), (340, 58), (50, 50, 50), -1)
        cv2.rectangle(frame, (160, 40), (160 + bar_w, 58), (0, 220, 120), -1)
        cv2.putText(frame, "hold", (345, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    panel_y = h - 200
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, panel_y), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay2, 0.82, frame, 0.18, 0, frame)

    current_word = "".join(word_buffer)
    cv2.putText(frame, f"Spelling:  {current_word}_", (15, panel_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2)

    if suggestions:
        cv2.putText(frame, "Suggestions:", (15, panel_y + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        sx = 130
        for i, sug in enumerate(suggestions):
            is_sel = (i == selected_suggest)
            color  = (0, 200, 255) if is_sel else (180, 180, 180)
            bg     = (40, 80, 100) if is_sel else (40, 40, 40)
            tw     = len(sug) * 11 + 16
            cv2.rectangle(frame, (sx - 4, panel_y + 48),
                          (sx + tw, panel_y + 72), bg, -1)
            cv2.putText(frame, f"{i+1}.{sug}", (sx, panel_y + 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            sx += tw + 12

    cv2.line(frame, (10, panel_y + 80),
             (w - 10, panel_y + 80), (60, 60, 60), 1)

    sent_str = "Sentence:  " + "  ›  ".join(sentence_words)
    cv2.putText(frame, sent_str[:68], (15, panel_y + 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 180, 255), 2)

    cv2.line(frame, (10, panel_y + 120),
             (w - 10, panel_y + 120), (60, 60, 60), 1)

    if show_sentence and final_sentence:
        cv2.rectangle(frame, (10, panel_y + 126),
                      (w - 10, panel_y + 168), (0, 50, 0), -1)
        cv2.putText(frame, f">> {final_sentence[:72]}", (18, panel_y + 153),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2)

    ctrl1 = "[M] Mode  [SPACE] Add letter/word  [1-4] Pick suggestion"
    ctrl2 = "[ENTER] Speak sentence  [BKSP] Undo  [C] Clear  [Q] Quit"
    cv2.putText(frame, ctrl1, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)
    cv2.putText(frame, ctrl2, (10, h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

    return frame

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

print("\n🚀 ISL Translator started!")
print("Controls:")
print("  [M]     — Toggle Letter / Word mode")
print("  [SPACE] — Confirm word (letter mode) / force-add prediction (word mode)")
print("  [1-4]   — Pick autocomplete suggestion")
print("  [ENTER] — Generate NLP sentence + speak")
print("  [BKSP]  — Undo last letter or word")
print("  [C]     — Clear everything")
print("  [Q]     — Quit\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detected   = ""
    confidence = 0.0

    # ── LETTER MODE — UNCHANGED ───────────────────────────────────────────────
    if mode == "LETTER":
        hand_results = hands.process(img_rgb)

        if hand_results.multi_hand_landmarks:
            for hl in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(
                        color=(255, 255, 0), thickness=2)
                )

            label, conf = predict_letter(hand_results)
            if conf >= LETTER_CONF_THRESH:
                detected   = label
                confidence = conf

                if label == last_letter:
                    letter_hold_count += 1
                else:
                    letter_hold_count = 0
                    last_letter = label

                if letter_hold_count >= LETTER_STABLE_FRAMES:
                    word_buffer.append(label.upper())
                    suggestions = get_autocomplete("".join(word_buffer))
                    print(f"🔤 Letter: {label}  |  Word so far: {''.join(word_buffer)}")
                    if suggestions:
                        print(f"   💡 Suggestions: {suggestions}")
                    letter_hold_count = 0
                    last_letter = ""
        else:
            letter_hold_count = 0

    # ── WORD MODE — FIXED ─────────────────────────────────────────────────────
    else:
        hol_results = holistic.process(img_rgb)

        if hol_results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hol_results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(
                    color=(255, 255, 0), thickness=2)
            )
        if hol_results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hol_results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(255, 100, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(
                    color=(255, 200, 0), thickness=2)
            )

        # ── FIX 1: skip everything if no hand is visible ──────────────────────
        if hand_visible(hol_results):
            frame_feats = extract_holistic_features(hol_results)  # (162,)

            # ── FIX 2: compute velocity (matches training data format) ─────────
            if prev_word_features is not None:
                velocity = frame_feats - prev_word_features
            else:
                velocity = np.zeros(162)
            prev_word_features = frame_feats.copy()

            combined = np.concatenate([frame_feats, velocity])    # (324,)
            frame_sequence.append(combined)
            word_frame_count += 1

            # ── FIX 3: predict every 5 frames once buffer is full ─────────────
            if len(frame_sequence) == SEQUENCE_LENGTH and word_frame_count % 5 == 0:
                label, conf = predict_word(list(frame_sequence))

                if conf >= WORD_CONF_THRESH:
                    detected   = label
                    confidence = conf
                    word_pred_history.append(label)

                    # ── FIX 4: require 5 consistent predictions to auto-confirm ─
                    if (len(word_pred_history) == WORD_STABLE_FRAMES and
                            len(set(word_pred_history)) == 1 and
                            label != last_word):
                        sentence_words.append(label)
                        last_word = label
                        word_pred_history.clear()
                        frame_sequence.clear()
                        print(f"📝 Word confirmed: {label}  ({conf:.0%})")
                else:
                    word_pred_history.clear()

        else:
            # ── FIX 1 continued: reset everything when hand leaves frame ──────
            frame_sequence.clear()
            prev_word_features = None
            word_pred_history.clear()
            word_frame_count = 0

    # ── DRAW UI ───────────────────────────────────────────────────────────────
    frame = draw_ui(
        frame, detected, confidence, mode,
        word_buffer, sentence_words,
        final_sentence, show_sentence,
        suggestions, selected_suggest,
        letter_hold_count
    )

    cv2.imshow("ISL Real-Time Translator", frame)

    # ── KEY HANDLING — UNCHANGED ──────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('m'):
        mode = "WORD" if mode == "LETTER" else "LETTER"
        frame_sequence.clear()
        word_frame_count   = 0
        letter_hold_count  = 0
        last_letter        = ""
        last_word          = ""
        suggestions        = []
        prev_word_features = None
        word_pred_history.clear()
        print(f"🔄 Mode: {mode}")

    elif key == ord('c'):
        word_buffer.clear()
        sentence_words.clear()
        final_sentence     = ""
        show_sentence      = False
        last_letter        = ""
        last_word          = ""
        letter_hold_count  = 0
        suggestions        = []
        selected_suggest   = -1
        frame_sequence.clear()
        prev_word_features = None
        word_pred_history.clear()
        print("🗑️  Cleared")

    elif key == 8:  # BACKSPACE
        if mode == "LETTER" and word_buffer:
            removed = word_buffer.pop()
            suggestions = get_autocomplete("".join(word_buffer))
            print(f"⬅️  Removed letter: {removed}")
        elif sentence_words:
            removed = sentence_words.pop()
            print(f"⬅️  Removed word: {removed}")

    elif key == 32:  # SPACE
        if mode == "LETTER":
            if word_buffer:
                word = "".join(word_buffer)
                sentence_words.append(word)
                print(f"📝 Word confirmed: {word}")
                word_buffer.clear()
                suggestions      = []
                selected_suggest = -1
            elif detected:
                word_buffer.append(detected.upper())
                suggestions = get_autocomplete("".join(word_buffer))
        elif mode == "WORD" and detected:
            # Manual override — SPACE force-adds current prediction
            if detected != last_word:
                sentence_words.append(detected)
                last_word = detected
                word_pred_history.clear()
                frame_sequence.clear()
                print(f"📝 Word force-added: {detected}")

    elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
        idx = key - ord('1')
        if idx < len(suggestions):
            chosen = suggestions[idx]
            sentence_words.append(chosen)
            print(f"✅ Autocomplete: {chosen}")
            word_buffer.clear()
            suggestions      = []
            selected_suggest = -1

    elif key == 13:  # ENTER
        if sentence_words:
            print(f"\n🔄 Generating: {sentence_words}")
            final_sentence = correct_sentence(sentence_words)
            show_sentence  = True
            print(f"✅ Final: {final_sentence}")
            threading.Thread(
                target=speak, args=(final_sentence,), daemon=True
            ).start()

cap.release()
cv2.destroyAllWindows()
print("👋 Closed.")