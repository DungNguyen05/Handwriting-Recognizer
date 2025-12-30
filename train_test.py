"""
EMNIST LETTER RECOGNITION - SIMPLE TRAINING
Sử dụng tensorflow_datasets để tải EMNIST
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Setup
np.random.seed(42)
tf.random.set_seed(42)
os.makedirs('models', exist_ok=True)

print("="*60)
print("EMNIST LETTER RECOGNITION - BASELINE MODEL")
print("="*60)

# Load EMNIST using tensorflow_datasets
print("\n📦 Loading EMNIST/balanced from tensorflow_datasets...")
print("(First time will download ~500MB)")

ds_train = tfds.load('emnist/balanced', split='train', as_supervised=True)
ds_test = tfds.load('emnist/balanced', split='test', as_supervised=True)

# Convert to numpy arrays
X_train_list, y_train_list = [], []
for image, label in tfds.as_numpy(ds_train):
    X_train_list.append(image)
    y_train_list.append(label)

X_test_list, y_test_list = [], []
for image, label in tfds.as_numpy(ds_test):
    X_test_list.append(image)
    y_test_list.append(label)

X_train = np.array(X_train_list)
y_train = np.array(y_train_list)
X_test = np.array(X_test_list)
y_test = np.array(y_test_list)

print(f"Raw data loaded: Train {X_train.shape}, Test {X_test.shape}")

# Filter letters only (remove digits 0-9)
train_mask = y_train >= 10
test_mask = y_test >= 10

X_train = X_train[train_mask]
y_train = y_train[train_mask] - 10  # Shift to 0-36

X_test = X_test[test_mask]
y_test = y_test[test_mask] - 10

# Preprocess
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 37)
y_test = to_categorical(y_test, 37)

print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
print(f"✅ Classes: 37 (26 uppercase + 11 lowercase)")

# Build Baseline Model
print("\n🔨 Building Baseline Model...")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(37, activation='softmax')
], name='Baseline')

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
print(f"\n📊 Total parameters: {model.count_params():,}")

# Train
print("\n🚀 Training...")
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ],
    verbose=1
)

# Evaluate
print("\n📈 Evaluating...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")

# Save
model_path = 'models/Baseline.h5'
model.save(model_path)
print(f"\n💾 Model saved to: {model_path}")

print("\n" + "="*60)
print("🎉 TRAINING COMPLETED!")
print("="*60)