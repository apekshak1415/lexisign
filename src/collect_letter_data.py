import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tqdm import tqdm

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,              # ← detect BOTH hands
    min_detection_confidence=0.3
)

DATA_DIR = './data/letters'
OUTPUT_FILE = './data/letter_landmarks.pkl'

data = []
labels = []
skipped = 0

classes = sorted(os.listdir(DATA_DIR))
print(f"Found {len(classes)} classes: {classes}")

def extract_two_hand_landmarks(results):
    """Extract landmarks for up to 2 hands, pad if only 1 or 0 detected."""
    all_landmarks = []

    detected_hands = results.multi_hand_landmarks or []

    for i in range(2):  # always produce exactly 2 hands worth of features
        if i < len(detected_hands):
            hl = detected_hands[i]
            x_coords = [lm.x for lm in hl.landmark]
            y_coords = [lm.y for lm in hl.landmark]
            min_x, min_y = min(x_coords), min(y_coords)
            for lm in hl.landmark:
                all_landmarks.append(lm.x - min_x)
                all_landmarks.append(lm.y - min_y)
        else:
            # Pad with zeros if hand not detected
            all_landmarks.extend([0.0] * 42)  # 21 landmarks * 2 (x,y)

    return all_landmarks  # always 84 features (2 hands * 21 * 2)

for label in tqdm(classes, desc="Processing classes"):
    class_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(class_path):
        continue

    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        img = cv2.imread(img_path)

        if img is None:
            skipped += 1
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            landmarks = extract_two_hand_landmarks(results)
            data.append(landmarks)
            labels.append(label)
        else:
            skipped += 1

print(f"\n✅ Processed {len(data)} images")
print(f"⚠️  Skipped {skipped} images (no hand detected)")
print(f"📁 Classes found: {len(set(labels))}")
print(f"🔢 Feature size per sample: {len(data[0]) if data else 0}")

os.makedirs('./data', exist_ok=True)
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"\n💾 Saved to {OUTPUT_FILE}")