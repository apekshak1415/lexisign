import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

print("=" * 50)
print("  LEXISIGN — Model Accuracy Check")
print("=" * 50)

# ── LETTER MODEL ──────────────────────────────────────
print("\n📂 Loading letter model and data...")
letter_model = load_model('./models/letter_model.h5')

with open('./models/label_encoder_letters.pkl', 'rb') as f:
    le_letters = pickle.load(f)

with open('./data/own_letter_landmarks.pkl', 'rb') as f:
    letter_data = pickle.load(f)

X_letters = np.array(letter_data['data'])
y_letters  = le_letters.transform(letter_data['labels'])

# Same split as training (random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X_letters, y_letters, test_size=0.2, random_state=42,
    stratify=y_letters
)

y_pred = np.argmax(letter_model.predict(X_test, verbose=0), axis=1)
letter_acc = accuracy_score(y_test, y_pred) * 100

print(f"\n✅ Letter Model:")
print(f"   Test samples : {len(X_test)}")
print(f"   Classes      : {len(le_letters.classes_)}")
print(f"   Accuracy     : {letter_acc:.2f}%")

# ── WORD MODEL ────────────────────────────────────────
print("\n📂 Loading word model and data...")
word_model = load_model('./models/word_model.h5')

with open('./models/label_encoder_words.pkl', 'rb') as f:
    le_words = pickle.load(f)

with open('./data/own_word_landmarks.pkl', 'rb') as f:
    word_data = pickle.load(f)

sequences = np.array(word_data['sequences'])
labels    = np.array(word_data['labels'])

# Only keep labels that exist in the trained encoder
known_mask = np.isin(labels, le_words.classes_)
sequences  = sequences[known_mask]
labels     = labels[known_mask]

if len(labels) == 0:
    print("   ⚠️  No matching labels found between data and encoder!")
else:
    y_encoded = le_words.transform(labels)
    y_cat     = to_categorical(y_encoded, num_classes=len(le_words.classes_))

    X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
        sequences, y_cat, test_size=0.15, random_state=42,
        stratify=y_encoded
    )

    y_pred_w   = word_model.predict(np.array(X_test_w, dtype=np.float32), verbose=0)
    y_pred_cls = np.argmax(y_pred_w, axis=1)
    y_true_cls = np.argmax(y_test_w, axis=1)
    word_acc   = accuracy_score(y_true_cls, y_pred_cls) * 100

    print(f"\n✅ Word Model:")
    print(f"   Test samples : {len(X_test_w)}")
    print(f"   Classes      : {len(le_words.classes_)}")
    print(f"   Classes list : {list(le_words.classes_)}")
    print(f"   Accuracy     : {word_acc:.2f}%")

# ── SUMMARY ───────────────────────────────────────────
print("\n" + "=" * 50)
print("  FINAL RESULTS (put these in your paper)")
print("=" * 50)
print(f"  Letter Model : {letter_acc:.2f}%")
try:
    print(f"  Word Model   : {word_acc:.2f}%")
except:
    print("  Word Model   : Could not evaluate (encoder mismatch)")
print("=" * 50)