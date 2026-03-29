import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tqdm import tqdm

mp_hands   = mp.solutions.hands
mp_pose    = mp.solutions.pose
mp_holistic = mp.solutions.holistic

# Use Holistic — detects hands + pose + face in one pass
holistic = mp_holistic.Holistic(
    static_image_mode=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

WORD_FRAMES_DIR = './data/words/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Word_Level'
OUTPUT_FILE     = './data/word_landmarks.pkl'
SEQUENCE_LENGTH = 10

data   = []
labels = []
skipped_words = []

word_classes = sorted(os.listdir(WORD_FRAMES_DIR))
print(f"Found {len(word_classes)} word classes")

def extract_holistic_features(img_rgb):
    """
    Extract full body features:
    - Left hand:  21 landmarks * 3 = 63
    - Right hand: 21 landmarks * 3 = 63
    - Pose upper body: 12 landmarks * 3 = 36
    Total: 162 features per frame
    """
    results = holistic.process(img_rgb)
    features = []

    # ── Right hand (63 features) ──────────────
    if results.right_hand_landmarks:
        rx = [lm.x for lm in results.right_hand_landmarks.landmark]
        ry = [lm.y for lm in results.right_hand_landmarks.landmark]
        min_rx, min_ry = min(rx), min(ry)
        for lm in results.right_hand_landmarks.landmark:
            features.extend([lm.x - min_rx, lm.y - min_ry, lm.z])
    else:
        features.extend([0.0] * 63)

    # ── Left hand (63 features) ───────────────
    if results.left_hand_landmarks:
        lx = [lm.x for lm in results.left_hand_landmarks.landmark]
        ly = [lm.y for lm in results.left_hand_landmarks.landmark]
        min_lx, min_ly = min(lx), min(ly)
        for lm in results.left_hand_landmarks.landmark:
            features.extend([lm.x - min_lx, lm.y - min_ly, lm.z])
    else:
        features.extend([0.0] * 63)

    # ── Pose upper body (36 features) ─────────
    # Indices: 0=nose,11=l_shoulder,12=r_shoulder,
    #          13=l_elbow,14=r_elbow,15=l_wrist,16=r_wrist
    #          23=l_hip,24=r_hip
    upper_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 1, 2, 4]
    if results.pose_landmarks:
        for idx in upper_indices:
            lm = results.pose_landmarks.landmark[idx]
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 36)

    return features  # 162 total

def pad_or_trim(sequence, length):
    if len(sequence) >= length:
        indices = np.linspace(0, len(sequence) - 1, length, dtype=int)
        return [sequence[i] for i in indices]
    while len(sequence) < length:
        sequence.append(sequence[-1])
    return sequence

for word in tqdm(word_classes, desc="Processing words"):
    word_path = os.path.join(WORD_FRAMES_DIR, word)
    if not os.path.isdir(word_path):
        continue

    frame_files = sorted([
        f for f in os.listdir(word_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not frame_files:
        skipped_words.append(word)
        continue

    sequence = []
    for ff in frame_files:
        img = cv2.imread(os.path.join(word_path, ff))
        if img is None:
            continue
        # Resize for faster processing
        img = cv2.resize(img, (640, 480))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        feats = extract_holistic_features(img_rgb)
        sequence.append(feats)

    if not sequence:
        skipped_words.append(word)
        continue

    sequence = pad_or_trim(sequence, SEQUENCE_LENGTH)
    data.append(sequence)
    labels.append(word)

print(f"\n✅ Processed {len(data)} word sequences")
print(f"⚠️  Skipped: {skipped_words}")
print(f"📁 Word classes: {len(set(labels))}")
print(f"🔢 Feature shape: {np.array(data).shape}")

os.makedirs('./data', exist_ok=True)
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"\n💾 Saved to {OUTPUT_FILE}")