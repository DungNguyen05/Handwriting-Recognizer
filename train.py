"""
EMNIST BALANCED RECOGNITION - OPTIMIZED FOR M1 SPEED
Updated: Using tf.data.Dataset pipeline for high-speed training
47 classes: A-Z (merged case) + 0-9
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers, models

# ============================================================
# M1/M2 MAC OPTIMIZATION SETUP
# ============================================================
def setup_m1_gpu():
    print("="*60)
    print("🍎 CHECKING APPLE SILICON GPU SETUP")
    print("="*60)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU DETECTED: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"   - {gpu}")
        
        try:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print("✅ Mixed Precision (float16) ACTIVATED for M1/M2")
        except Exception as e:
            print(f"⚠️ Could not set mixed precision: {e}")
            
    else:
        print("❌ NO GPU DETECTED. Training will default to CPU.")

setup_m1_gpu()
np.random.seed(42)
tf.random.set_seed(42)
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# ============================================================
# MODEL ARCHITECTURES
# ============================================================

def build_baseline_model():
    """BASELINE - Simple CNN"""
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(47, activation='softmax', dtype='float32') 
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
    
    # Residual Block 1
    shortcut = x
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Residual Block 2
    shortcut = layers.Conv2D(64, (1, 1), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Residual Block 3
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
    outputs = layers.Dense(47, activation='softmax', dtype='float32')(x)
    
    model = models.Model(inputs, outputs, name='ResNet')
    return model

# ============================================================
# DATA LOADING
# ============================================================

def get_optimized_datasets(batch_size=256):
    """
    Load EMNIST balanced using tf.data pipeline for max speed on M1
    47 classes: A-Z (merged case) + 0-9
    """
    print("\n📦 Loading EMNIST/balanced using Optimized tf.data Pipeline...")
    
    # Load raw data
    (ds_train_full, ds_test), ds_info = tfds.load(
        'emnist/balanced', 
        split=['train', 'test'], 
        as_supervised=True, 
        with_info=True
    )

    # Hàm chuẩn hóa và xoay ảnh (chạy song song trong pipeline)
    def preprocess(image, label):
        # Fix EMNIST rotation (Transpose dimensions)
        image = tf.transpose(image, perm=[1, 0, 2])
        # Normalize to 0-1
        image = tf.cast(image, tf.float32) / 255.0
        # One-hot encoding
        label = tf.one_hot(label, 47)
        return image, label

    AUTOTUNE = tf.data.AUTOTUNE
    
    # --- SETUP TRAINING DATA (Split Train/Val thủ công vì dùng Dataset) ---
    TOTAL_TRAIN = ds_info.splits['train'].num_examples
    VAL_SIZE = int(TOTAL_TRAIN * 0.1) 
    TRAIN_SIZE = TOTAL_TRAIN - VAL_SIZE
    
    print(f"   • Total Train Samples: {TOTAL_TRAIN}")
    print(f"   • Splitting: {TRAIN_SIZE} Train / {VAL_SIZE} Validation")

    ds_train_full = ds_train_full.map(preprocess, num_parallel_calls=AUTOTUNE)
    ds_train_full = ds_train_full.cache() # Cache vào RAM để đọc siêu nhanh
    ds_train_full = ds_train_full.shuffle(TOTAL_TRAIN)

    # Chia Train / Val
    ds_train = ds_train_full.take(TRAIN_SIZE)
    ds_val   = ds_train_full.skip(TRAIN_SIZE)

    # Batching & Prefetching
    ds_train = ds_train.batch(batch_size).prefetch(AUTOTUNE)
    ds_val   = ds_val.batch(batch_size).prefetch(AUTOTUNE)

    # --- SETUP TEST DATA ---
    ds_test = ds_test.map(preprocess, num_parallel_calls=AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache().prefetch(AUTOTUNE)

    return ds_train, ds_val, ds_test

# ============================================================
# VISUALIZATION
# ============================================================

def plot_training_history(history, model_name, save_path='plots'):
    """Visualize training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_name.upper()} - Training History', fontsize=16, fontweight='bold')
    
    # 1. Accuracy
    ax1 = axes[0, 0]
    ax1.plot(history.history['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss
    ax2 = axes[0, 1]
    ax2.plot(history.history['loss'], 'b-', label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Overfitting Gap
    ax3 = axes[1, 0]
    acc_gap = np.array(history.history['accuracy']) - np.array(history.history['val_accuracy'])
    ax3.plot(acc_gap, 'g-', label='Train - Val Gap', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_title('Overfitting Monitor (Accuracy Gap)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary Text
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"Final Val Acc: {history.history['val_accuracy'][-1]:.4f}\nMin Val Loss: {min(history.history['val_loss']):.4f}"
    ax4.text(0.1, 0.5, summary_text, fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, f'{model_name}_training_history.png')
    plt.savefig(plot_path)
    print(f"\n📊 Plot saved to: {plot_path}")
    # plt.show() # Commented out for non-interactive envs, uncomment if needed

def print_training_summary(history, model_name, test_acc, training_time):
    print("\n" + "="*60)
    print(f"📈 {model_name.upper()} - TRAINING SUMMARY")
    print("="*60)
    print(f"  Final Train Acc: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"  Final Val Acc:   {history.history['val_accuracy'][-1]*100:.2f}%")
    print(f"  Test Accuracy:   {test_acc*100:.2f}%")
    print(f"  Training Time:   {training_time:.2f}s")
    print("="*60)

# ============================================================
# TRAINING
# ============================================================

def train_model(model_name, epochs=15, batch_size=256):
    print("="*60)
    print(f"EMNIST BALANCED RECOGNITION - {model_name.upper()} MODEL")
    print("="*60)
    
    # 1. LOAD DATASET (OPTIMIZED)
    ds_train, ds_val, ds_test = get_optimized_datasets(batch_size=batch_size)
    
    # 2. Build model
    print(f"\n🔨 Building {model_name.upper()} Model...")
    if model_name.lower() == 'baseline':
        model = build_baseline_model()
    elif model_name.lower() == 'resnet':
        model = build_resnet_deep_model()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    
    # 3. TRAIN
    print("\n🚀 Training (High Performance Mode)...")
    start_time = time.time()
    
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # 4. Evaluate
    print("\n📈 Evaluating on test set...")
    test_loss, test_acc = model.evaluate(ds_test, verbose=0)
    
    # Print summary & Plot
    print_training_summary(history, model_name, test_acc, training_time)
    plot_training_history(history, model_name)
    
    # Save
    model_path = f"models/{model_name.capitalize()}.keras"
    model.save(model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    return model, history

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    # Tăng batch_size lên 256 để tận dụng GPU M1
    train_model(
        model_name='baseline',
        epochs=20,
        batch_size=256 
    )