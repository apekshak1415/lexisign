import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

print("📂 Loading YOUR collected data...")
with open('./data/own_letter_landmarks.pkl', 'rb') as f:
    dataset = pickle.load(f)

data   = np.array(dataset['data'])
labels = np.array(dataset['labels'])

print(f"✅ Total samples : {len(data)}")
print(f"🔢 Feature size  : {data.shape[1]}")
print(f"📊 Classes       : {sorted(set(labels))}")

# ── Label encoding ────────────────────────────
le = LabelEncoder()
labels_encoded     = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

os.makedirs('./models', exist_ok=True)
with open('./models/label_encoder_letters.pkl', 'wb') as f:
    pickle.dump(le, f)
print(f"\n🏷️  Classes: {list(le.classes_)}")

# ── Data augmentation ─────────────────────────
print("\n🔄 Augmenting data...")

def augment(sample, n=3):
    augmented = []
    for _ in range(n):
        s = sample.copy()
        s += np.random.normal(0, 0.005, s.shape)   # tiny noise
        s *= np.random.uniform(0.95, 1.05)           # tiny scale
        augmented.append(s)
    return augmented

aug_data, aug_labels = [], []
for i in range(len(data)):
    aug_data.append(data[i])
    aug_labels.append(labels_encoded[i])
    for a in augment(data[i], n=3):
        aug_data.append(a)
        aug_labels.append(labels_encoded[i])

aug_data   = np.array(aug_data)
aug_labels = np.array(aug_labels)
aug_labels_cat = to_categorical(aug_labels)

print(f"✅ Augmented size: {len(aug_data)}")

# ── Split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    aug_data, aug_labels_cat,
    test_size=0.15,
    random_state=42,
    stratify=aug_labels
)
print(f"📈 Train: {len(X_train)}  |  📉 Test: {len(X_test)}")

num_classes  = len(le.classes_)
feature_size = aug_data.shape[1]

# ── Model ─────────────────────────────────────
model = Sequential([
    Dense(512, activation='relu', input_shape=(feature_size,)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── Train ─────────────────────────────────────
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint('./models/letter_model.h5',
                    save_best_only=True, verbose=1),
]

print("\n🚀 Training started...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ── Evaluate ──────────────────────────────────
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# ── Plot ──────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],     label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'],     label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('./models/own_letter_training_plot.png')
plt.show()
print("📈 Plot saved to ./models/own_letter_training_plot.png")