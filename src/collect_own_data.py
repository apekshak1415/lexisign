import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import time

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

# Define which signs to collect
# Start with numbers and a few letters
SIGNS_TO_COLLECT = [
    '0','1','2','3','4','5','6','7','8','9',
    'a','b','c','d','e','f','g','h','i','j','k','l',
    'm','n','o','p','q','r','s','t','u','v','w','x','y','z'
]

SAMPLES_PER_SIGN = 200   # 200 samples per sign
SAVE_DIR         = './data/own_letters'
os.makedirs(SAVE_DIR, exist_ok=True)

def extract_two_hand_landmarks(results):
    features = []
    detected = results.multi_hand_landmarks or []
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

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

all_data   = []
all_labels = []

# Load existing data if any
existing_file = './data/own_letter_landmarks.pkl'
if os.path.exists(existing_file):
    with open(existing_file, 'rb') as f:
        existing = pickle.load(f)
    all_data   = existing['data']
    all_labels = existing['labels']
    print(f"📂 Loaded {len(all_data)} existing samples")

print("\n🎯 ISL Data Collector")
print("=" * 50)
print("Instructions:")
print("  - Show the sign clearly to the camera")
print("  - Keep your hand(s) FULLY in frame")
print("  - Press SPACE to start collecting")
print("  - Stay still while collecting")
print("  - Press S to skip a sign")
print("  - Press Q to quit and save")
print("=" * 50)

for sign in SIGNS_TO_COLLECT:
    # Check how many samples already collected
    existing_count = all_labels.count(sign)
    if existing_count >= SAMPLES_PER_SIGN:
        print(f"✅ {sign}: already has {existing_count} samples, skipping")
        continue

    needed = SAMPLES_PER_SIGN - existing_count
    collected = 0
    collecting = False
    countdown  = 0

    print(f"\n👉 Next sign: [{sign.upper()}]  (need {needed} more samples)")

    while collected < needed:
        ret, frame = cap.read()
        if not ret:
            break

        frame    = cv2.flip(frame, 1)
        img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results  = hands.process(img_rgb)

        # Draw landmarks
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255,255,0), thickness=2)
                )

        # UI
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0,0), (w, 80), (20,20,20), -1)

        # Sign label
        cv2.putText(frame, f"Sign: [{sign.upper()}]", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 200) if not collecting else (0, 255, 0), 3)

        # Progress bar
        progress = int((collected / needed) * (w - 40))
        cv2.rectangle(frame, (20, h-40), (w-20, h-15), (50,50,50), -1)
        cv2.rectangle(frame, (20, h-40), (20+progress, h-15), (0,200,100), -1)
        cv2.putText(frame, f"{collected}/{needed}", (w//2 - 40, h-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        if not collecting:
            cv2.putText(frame, "SPACE=Start  S=Skip  Q=Quit",
                        (20, h-55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (180,180,180), 1)
        else:
            cv2.putText(frame, "COLLECTING...", (w-220, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Hand detection indicator
        hand_color = (0,255,0) if results.multi_hand_landmarks else (0,0,255)
        hand_text  = f"Hands: {len(results.multi_hand_landmarks or [])}"
        cv2.putText(frame, hand_text, (w-180, h-55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)

        cv2.imshow("ISL Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # Save and quit
            with open(existing_file, 'wb') as f:
                pickle.dump({'data': all_data, 'labels': all_labels}, f)
            print(f"\n💾 Saved {len(all_data)} total samples")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        elif key == ord('s'):
            print(f"⏭️  Skipped [{sign}]")
            break

        elif key == 32:  # SPACE
            if results.multi_hand_landmarks:
                collecting = True
            else:
                print("⚠️  No hand detected! Show your hand first.")

        # Collect sample
        if collecting and results.multi_hand_landmarks:
            feats = extract_two_hand_landmarks(results)
            all_data.append(feats)
            all_labels.append(sign)
            collected += 1

            # Small pause every 50 samples
            if collected % 50 == 0:
                print(f"  📸 {collected}/{needed} for [{sign}]")
                collecting = False
                time.sleep(0.3)
                collecting = True

        elif collecting and not results.multi_hand_landmarks:
            collecting = False
            print("⚠️  Hand lost! Press SPACE to resume.")

    print(f"✅ Collected {collected} samples for [{sign}]")

# Final save
with open(existing_file, 'wb') as f:
    pickle.dump({'data': all_data, 'labels': all_labels}, f)

print(f"\n💾 Final save: {len(all_data)} total samples")
print("🎉 Data collection complete!")
cap.release()
cv2.destroyAllWindows()