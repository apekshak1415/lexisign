import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time

# ─── CONFIG ───────────────────────────────────────────────────────────────────
WORDS = [
    "HELLO", "THANK YOU", "SORRY", "HELP", "YES",
    "NO", "FOOD", "WATER", "GOOD", "MY NAME",
    "GOODBYE", "GOOD MORNING", "GOOD AFTERNOON", "GOOD EVENING",
    "HOW ARE YOU", "I AM FINE", "NICE TO MEET YOU",
    "SEE YOU LATER", "WELCOME", "I LOVE YOU", "GOOD NIGHT", "WHAT","WHERE","HOW","WHEN"
]
FRAMES_PER_CLIP  = 30      # frames captured per sign recording
CLIPS_PER_WORD   = 30      # how many times you sign each word
SAVE_PATH        = "./data/own_word_landmarks.pkl"

# ─── MEDIAPIPE SETUP ──────────────────────────────────────────────────────────
mp_holistic   = mp.solutions.holistic
mp_drawing    = mp.solutions.drawing_utils

POSE_LANDMARKS_USED = [11,12,13,14,15,16,23,24,25,26,27,28]  # upper body only

def extract_features(results):
    """
    Extract 162 features from holistic results:
      - Right hand : 21 landmarks x 3 = 63
      - Left hand  : 21 landmarks x 3 = 63
      - Pose (upper): 12 landmarks x 3 = 36
    Returns a numpy array of shape (162,).
    Zero-padded if hand not detected.
    """
    def hand_landmarks(hand):
        if hand:
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
            # normalize xy relative to wrist
            pts[0::3] -= pts[0]
            pts[1::3] -= pts[1]
            return pts
        return np.zeros(63)

    def pose_landmarks(pose):
        if pose:
            pts = np.array([[pose.landmark[i].x,
                             pose.landmark[i].y,
                             pose.landmark[i].z] for i in POSE_LANDMARKS_USED]).flatten()
            # normalize relative to shoulder midpoint
            sx = (pose.landmark[11].x + pose.landmark[12].x) / 2
            sy = (pose.landmark[11].y + pose.landmark[12].y) / 2
            pts[0::3] -= sx
            pts[1::3] -= sy
            return pts
        return np.zeros(36)

    rh = hand_landmarks(results.right_hand_landmarks)
    lh = hand_landmarks(results.left_hand_landmarks)
    ps = pose_landmarks(results.pose_landmarks)
    return np.concatenate([rh, lh, ps])  # shape (162,)


def add_velocity(sequence):
    """
    Given sequence of shape (T, 162), compute frame-to-frame deltas
    and concatenate → output shape (T, 324).
    First frame delta = zeros.
    """
    deltas = np.zeros_like(sequence)
    deltas[1:] = sequence[1:] - sequence[:-1]
    return np.concatenate([sequence, deltas], axis=1)  # (T, 324)


def collect_clip(cap, holistic, word, clip_num, total_clips):
    """
    Collect one clip of FRAMES_PER_CLIP frames.
    Returns numpy array of shape (FRAMES_PER_CLIP, 324) or None if cancelled.
    """
    frames_data = []

    # ── Countdown before recording ────────────────────────────────────────────
    for countdown in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Word: {word}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(frame, f"Clip {clip_num}/{total_clips}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
        cv2.putText(frame, f"Get ready... {countdown}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
        cv2.imshow("ISL Word Data Collection", frame)
        cv2.waitKey(700)

    # ── Record FRAMES_PER_CLIP frames ─────────────────────────────────────────
    start_time = time.time()
    while len(frames_data) < FRAMES_PER_CLIP:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        features = extract_features(results)
        frames_data.append(features)

        # Draw landmarks on screen
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS)

        progress = len(frames_data) / FRAMES_PER_CLIP
        bar_w = int(600 * progress)
        cv2.rectangle(frame, (20, 450), (620, 470), (60,60,60), -1)
        cv2.rectangle(frame, (20, 450), (20+bar_w, 470), (0,220,100), -1)
        cv2.putText(frame, f"RECORDING  {word}", (20, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,100), 2)
        cv2.putText(frame, f"Clip {clip_num}/{total_clips}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
        cv2.imshow("ISL Word Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None

    if len(frames_data) < FRAMES_PER_CLIP:
        return None

    sequence = np.array(frames_data)             # (30, 162)
    sequence = add_velocity(sequence)            # (30, 324)
    return sequence


def augment_sequence(seq, n=7):
    """
    Generate n augmented versions of a sequence.
    Augmentations: noise, scale, time warp, flip left/right hand.
    """
    augmented = []
    for _ in range(n):
        aug = seq.copy()
        # Gaussian noise
        aug += np.random.normal(0, 0.005, aug.shape)
        # Random scale
        scale = np.random.uniform(0.9, 1.1)
        aug  *= scale
        # Time warp: randomly drop and repeat 1 frame
        if np.random.rand() > 0.5:
            idx = np.random.randint(1, len(aug)-1)
            aug = np.delete(aug, idx, axis=0)
            aug = np.insert(aug, idx, aug[idx], axis=0)
        augmented.append(aug)
    return augmented


def main():
    # Load existing data if any
    all_sequences = []
    all_labels    = []
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "rb") as f:
            saved = pickle.load(f)
        all_sequences = list(saved["sequences"])
        all_labels    = list(saved["labels"])
        print(f"Loaded {len(all_sequences)} existing clips.")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    with mp_holistic.Holistic(min_detection_confidence=0.6,
                               min_tracking_confidence=0.6) as holistic:

        for word in WORDS:
            # Count how many clips we already have for this word
            existing = sum(1 for l in all_labels if l == word)
            if existing >= CLIPS_PER_WORD:
                print(f"[SKIP] {word} already has {existing} clips.")
                continue

            clips_needed = CLIPS_PER_WORD - existing
            print(f"\n{'='*50}")
            print(f"  Word: {word}  |  Need {clips_needed} more clips")
            print(f"{'='*50}")

            # Show instruction screen
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                cv2.rectangle(frame, (0,0), (960,540), (30,30,30), -1)
                cv2.putText(frame, f"Next word: {word}", (50, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255,255,255), 3)
                cv2.putText(frame, f"You will sign it {clips_needed} times.", (50, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180,180,180), 2)
                cv2.putText(frame, "Press SPACE to start   |   Q to quit", (50, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,220,255), 2)
                cv2.imshow("ISL Word Data Collection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    break
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # Collect clips for this word
            clip_count = existing
            while clip_count < CLIPS_PER_WORD:
                # Wait for SPACE press to record next clip
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(rgb)
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
                                              mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,
                                              mp_holistic.HAND_CONNECTIONS)
                    cv2.putText(frame, f"{word}  —  clip {clip_count+1}/{CLIPS_PER_WORD}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                    cv2.putText(frame, "SPACE = record next clip   |   N = next word   |   Q = quit",
                                (20, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
                    cv2.imshow("ISL Word Data Collection", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):
                        break
                    if key == ord('n'):
                        clip_count = CLIPS_PER_WORD  # skip to next word
                        break
                    if key == ord('q'):
                        # Save and exit
                        _save(all_sequences, all_labels)
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                if clip_count >= CLIPS_PER_WORD:
                    break

                seq = collect_clip(cap, holistic, word, clip_count+1, CLIPS_PER_WORD)
                if seq is None:
                    print("Recording cancelled.")
                    break

                # Save original + augmented
                all_sequences.append(seq)
                all_labels.append(word)
                augmented = augment_sequence(seq, n=7)
                for aug in augmented:
                    all_sequences.append(aug)
                    all_labels.append(word)

                clip_count += 1
                print(f"  ✓ Clip {clip_count}/{CLIPS_PER_WORD} saved  "
                      f"(+7 augmented = {clip_count*8} total for {word})")

                # Auto-save after every clip
                _save(all_sequences, all_labels)

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Collection complete!")
    _print_summary(all_labels)


def _save(sequences, labels):
    os.makedirs("./data", exist_ok=True)
    with open(SAVE_PATH, "wb") as f:
        pickle.dump({"sequences": np.array(sequences),
                     "labels":    np.array(labels)}, f)
    print(f"  💾 Saved {len(sequences)} clips to {SAVE_PATH}")


def _print_summary(labels):
    from collections import Counter
    counts = Counter(labels)
    print("\n── Dataset summary ──────────────────────")
    for word, count in sorted(counts.items()):
        print(f"  {word:<15} {count:>4} clips")
    print(f"  {'TOTAL':<15} {len(labels):>4} clips")
    print("─────────────────────────────────────────")


if __name__ == "__main__":
    main()