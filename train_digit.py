"""
EMNIST DIGIT RECOGNITION - TRAINING WITH MULTIPLE MODELS
Chỉ train với chữ số 0-9
Hỗ trợ 2 models: Baseline và ResNet Deep
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

# ============================================================
# MODEL ARCHITECTURES
# ============================================================

def build_baseline_model():
    """BASELINE - Simple CNN"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ], name='Baseline')
    
    return model


def build_resnet_deep_model():
    """RESNET-DEEPER - 3 Residual blocks"""
    inputs = layers.Input(shape=(28, 28, 1))
    
    x = inputs
    
    # Initial conv
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Residual Block 1 (32 filters)
    shortcut = x
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Residual Block 2 (64 filters)
    shortcut = layers.Conv2D(64, (1, 1), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Residual Block 3 (128 filters)
    shortcut = layers.Conv2D(128, (1, 1), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Dense
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name='ResNet')
    return model


# ============================================================
# DATA LOADING
# ============================================================

def load_emnist_digits():
    """Load and preprocess EMNIST digits dataset"""
    print("\n📦 Loading EMNIST/digits from tensorflow_datasets...")
    print("(First time will download data)")

    ds_train = tfds.load('emnist/digits', split='train', as_supervised=True)
    ds_test = tfds.load('emnist/digits', split='test', as_supervised=True)

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

    # Preprocess
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"✅ Classes: 10 (digits 0-9)")
    
    return X_train, y_train, X_test, y_test


# ============================================================
# TRAINING
# ============================================================

def train_model(model_name, epochs=15, batch_size=128):
    """Train specified model"""
    
    print("="*60)
    print(f"EMNIST DIGIT RECOGNITION - {model_name.upper()} MODEL")
    print("="*60)
    
    # Load data
    X_train, y_train, X_test, y_test = load_emnist_digits()
    
    # Build model
    print(f"\n🔨 Building {model_name.upper()} Model...")
    if model_name.lower() == 'baseline':
        model = build_baseline_model()
    elif model_name.lower() == 'resnet':
        model = build_resnet_deep_model()
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'baseline' or 'resnet'")
    
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
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ],
        verbose=1
    )
    
    # Evaluate
    print("\n📈 Evaluating...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
    print(f"✅ Test Loss: {test_loss:.4f}")
    
    # Save
    model_filename = f"{model_name.capitalize()}.h5"
    model_path = f"models/{model_filename}"
    model.save(model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED!")
    print("="*60)
    
    return model, history


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    # Thay đổi tên model ở đây: 'baseline' hoặc 'resnet'
    MODEL_NAME = 'resnet'
    
    train_model(
        model_name=MODEL_NAME,
        epochs=5,
        batch_size=128
    )