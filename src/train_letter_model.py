import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load data
print("📂 Loading landmark data...")
with open('./data/letter_landmarks.pkl', 'rb') as f:
    dataset = pickle.load(f)

data = np.array(dataset['data'])
labels = np.array(dataset['labels'])

print(f"✅ Total samples: {len(data)}")
print(f"📊 Classes: {sorted(set(labels))}")
print(f"🔢 Feature size: {data.shape[1]}")

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Save label encoder
import pickle
with open('./models/label_encoder_letters.pkl', 'wb') as f:
    pickle.dump(le, f)

print(f"\n🏷️  Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    data, labels_categorical, test_size=0.2, random_state=42, stratify=labels_encoded
)

print(f"\n📈 Training samples: {len(X_train)}")
print(f"📉 Testing samples:  {len(X_test)}")

num_classes = len(le.classes_)

# Build Model
model = Sequential([
    Dense(256, activation='relu', input_shape=(data.shape[1],)),
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
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
os.makedirs('./models', exist_ok=True)

callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint('./models/letter_model.h5', save_best_only=True, verbose=1)
]

# Train
print("\n🚀 Training started...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
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
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('./models/letter_training_plot.png')
plt.show()
print("\n📈 Training plot saved to ./models/letter_training_plot.png")