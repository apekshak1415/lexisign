import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

print("📂 Loading word landmark data...")
with open('./data/word_landmarks.pkl', 'rb') as f:
    dataset = pickle.load(f)

data = np.array(dataset['data'], dtype=np.float32)
labels = np.array(dataset['labels'])

print(f"✅ Total samples: {len(data)}")
print(f"🔢 Shape: {data.shape}")
print(f"📊 Classes: {len(set(labels))}")

# With only 114 samples (1 per word), we need data augmentation
print("\n🔄 Augmenting data...")

def augment_sequence(sequence, n_augments=20):
    augmented = []
    for _ in range(n_augments):
        aug = sequence.copy()
        # Add small random noise
        noise = np.random.normal(0, 0.01, aug.shape)
        aug = aug + noise
        # Random scaling
        scale = np.random.uniform(0.9, 1.1)
        aug = aug * scale
        # Random time shift
        shift = np.random.randint(-2, 3)
        aug = np.roll(aug, shift, axis=0)
        augmented.append(aug)
    return augmented

augmented_data = []
augmented_labels = []

for i in range(len(data)):
    # Keep original
    augmented_data.append(data[i])
    augmented_labels.append(labels[i])
    # Add augmented versions
    augs = augment_sequence(data[i], n_augments=25)
    augmented_data.extend(augs)
    augmented_labels.extend([labels[i]] * 25)

data = np.array(augmented_data, dtype=np.float32)
labels = np.array(augmented_labels)

print(f"✅ Augmented dataset size: {len(data)}")

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

os.makedirs('./models', exist_ok=True)
with open('./models/label_encoder_words.pkl', 'wb') as f:
    pickle.dump(le, f)

print(f"🏷️  Saved label encoder with {len(le.classes_)} classes")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    data, labels_categorical,
    test_size=0.2,
    random_state=42,
    stratify=labels_encoded
)

print(f"\n📈 Training samples: {len(X_train)}")
print(f"📉 Testing samples:  {len(X_test)}")

num_classes = len(le.classes_)
sequence_length = data.shape[1]
feature_size = data.shape[2]

# Build Bidirectional LSTM Model
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True),
                  input_shape=(sequence_length, feature_size)),
    BatchNormalization(),
    Dropout(0.3),

    Bidirectional(LSTM(64, return_sequences=False)),
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

# Callbacks
callbacks = [
    EarlyStopping(patience=20, restore_best_weights=True, verbose=1),
    ModelCheckpoint('./models/word_model.h5', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                      min_lr=1e-6, verbose=1)
]

# Train
print("\n🚀 Training started...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
print("\n📊 Evaluating model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")

# Classification Report
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# Plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Word Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Word Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('./models/word_training_plot.png')
plt.show()
print("\n📈 Training plot saved to ./models/word_training_plot.png")