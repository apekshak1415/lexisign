import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing     import LabelEncoder
from sklearn.metrics           import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models     import Sequential
from tensorflow.keras.layers     import (Bidirectional, LSTM, Dense,
                                          Dropout, BatchNormalization)
from tensorflow.keras.callbacks  import (EarlyStopping, ReduceLROnPlateau,
                                          ModelCheckpoint)
from tensorflow.keras.utils      import to_categorical

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH        = "./data/own_word_landmarks.pkl"
MODEL_PATH       = "./models/word_model.h5"
ENCODER_PATH     = "./models/label_encoder_words.pkl"
PLOT_PATH        = "./models/word_training_plot.png"

SEQUENCE_LEN     = 30    # frames per clip
FEATURE_DIM      = 324   # 162 landmarks + 162 velocity deltas
TEST_SIZE        = 0.15
RANDOM_STATE     = 42

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
print("Loading data...")
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

sequences = data["sequences"]   # (N, 30, 324)
labels    = data["labels"]      # (N,)

print(f"  Total clips : {len(sequences)}")
print(f"  Sequence shape: {sequences[0].shape}")

from collections import Counter
counts = Counter(labels)
print("\n── Clips per word ───────────────────────")
for word, count in sorted(counts.items()):
    print(f"  {word:<15} {count:>4}")
print("─────────────────────────────────────────\n")

# ─── ENCODE LABELS ────────────────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(labels)
num_classes = len(le.classes_)
print(f"Classes ({num_classes}): {list(le.classes_)}\n")

y_cat = to_categorical(y_encoded, num_classes=num_classes)

# Verify / fix sequence length
X = []
for seq in sequences:
    if len(seq) >= SEQUENCE_LEN:
        X.append(seq[:SEQUENCE_LEN])
    else:
        # Pad with last frame if too short
        pad = np.tile(seq[-1], (SEQUENCE_LEN - len(seq), 1))
        X.append(np.vstack([seq, pad]))

X = np.array(X, dtype=np.float32)   # (N, 30, 324)
y = y_cat

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# ─── TRAIN / TEST SPLIT ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
)
print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}\n")

# ─── MODEL ────────────────────────────────────────────────────────────────────
def build_model(seq_len, feature_dim, num_classes):
    model = Sequential([
        # Layer 1 — Bi-LSTM reads full sequence both ways
        Bidirectional(
            LSTM(128, return_sequences=True),
            input_shape=(seq_len, feature_dim)
        ),
        BatchNormalization(),
        Dropout(0.3),

        # Layer 2 — second Bi-LSTM distills motion context
        Bidirectional(LSTM(64, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.3),

        # Dense layers for classification
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dropout(0.2),

        Dense(num_classes, activation="softmax")
    ])
    return model

model = build_model(SEQUENCE_LEN, FEATURE_DIM, num_classes)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ─── CALLBACKS ────────────────────────────────────────────────────────────────
os.makedirs("./models", exist_ok=True)

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=20,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=10, min_lr=1e-6, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                    save_best_only=True, verbose=1)
]

# ─── TRAIN ────────────────────────────────────────────────────────────────────
print("\nTraining...\n")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=150,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# ─── EVALUATE ─────────────────────────────────────────────────────────────────
print("\n── Final Evaluation ─────────────────────")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"  Test accuracy : {acc*100:.2f}%")
print(f"  Test loss     : {loss:.4f}")

y_pred    = model.predict(X_test, verbose=0)
y_pred_cl = np.argmax(y_pred, axis=1)
y_true_cl = np.argmax(y_test, axis=1)

print("\n── Classification Report ────────────────")
print(classification_report(y_true_cl, y_pred_cl,
                             target_names=le.classes_))

# ─── SAVE ENCODER ─────────────────────────────────────────────────────────────
with open(ENCODER_PATH, "wb") as f:
    pickle.dump(le, f)
print(f"\n✅ Model saved  : {MODEL_PATH}")
print(f"✅ Encoder saved: {ENCODER_PATH}")

# ─── PLOT ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["accuracy"],     label="Train acc")
axes[0].plot(history.history["val_accuracy"], label="Val acc")
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["loss"],     label="Train loss")
axes[1].plot(history.history["val_loss"], label="Val loss")
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=120)
print(f"✅ Plot saved   : {PLOT_PATH}")
plt.show()

# Confusion matrix
cm = confusion_matrix(y_true_cl, y_pred_cl)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix — Word Model")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig("./models/word_confusion_matrix.png", dpi=120)
plt.show()
print("✅ Confusion matrix saved.")