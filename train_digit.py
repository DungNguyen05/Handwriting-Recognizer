"""
EMNIST DIGIT RECOGNITION - TRAINING WITH VISUALIZATION
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

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
            print(f"   - Compute dtype: {policy.compute_dtype}")
            print(f"   - Variable dtype: {policy.variable_dtype}")
        except Exception as e:
            print(f"⚠️ Could not set mixed precision: {e}")
            
    else:
        print("❌ NO GPU DETECTED. Training will default to CPU.")
        print("   Make sure you installed 'tensorflow-metal'")

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
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax', dtype='float32') 
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
    outputs = layers.Dense(10, activation='softmax', dtype='float32')(x)
    
    model = models.Model(inputs, outputs, name='ResNet')
    return model


# ============================================================
# DATA LOADING
# ============================================================

def load_emnist_digits():
    """Load and preprocess EMNIST digits dataset"""
    print("\n📦 Loading EMNIST/digits from tensorflow_datasets...")
    
    ds_train = tfds.load('emnist/digits', split='train', as_supervised=True, download=True)
    ds_test = tfds.load('emnist/digits', split='test', as_supervised=True, download=True)

    X_train = np.array([x for x, y in tfds.as_numpy(ds_train)])
    y_train = np.array([y for x, y in tfds.as_numpy(ds_train)])
    
    X_test = np.array([x for x, y in tfds.as_numpy(ds_test)])
    y_test = np.array([y for x, y in tfds.as_numpy(ds_test)])

    print(f"Raw data loaded: Train {X_train.shape}, Test {X_test.shape}")

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test


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
    ax1.set_title('Model Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Hiển thị max accuracy
    max_train = max(history.history['accuracy'])
    max_val = max(history.history['val_accuracy'])
    ax1.axhline(y=max_train, color='b', linestyle='--', alpha=0.3)
    ax1.axhline(y=max_val, color='r', linestyle='--', alpha=0.3)
    ax1.text(0.02, 0.98, f'Max Train: {max_train:.4f}\nMax Val: {max_val:.4f}', 
             transform=ax1.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Loss
    ax2 = axes[0, 1]
    ax2.plot(history.history['loss'], 'b-', label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Hiển thị min loss
    min_train = min(history.history['loss'])
    min_val = min(history.history['val_loss'])
    ax2.axhline(y=min_train, color='b', linestyle='--', alpha=0.3)
    ax2.axhline(y=min_val, color='r', linestyle='--', alpha=0.3)
    ax2.text(0.02, 0.98, f'Min Train: {min_train:.4f}\nMin Val: {min_val:.4f}', 
             transform=ax2.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Overfitting Analysis
    ax3 = axes[1, 0]
    acc_gap = np.array(history.history['accuracy']) - np.array(history.history['val_accuracy'])
    ax3.plot(acc_gap, 'g-', label='Train - Val Gap', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_title('Overfitting Monitor (Accuracy Gap)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy Gap')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.fill_between(range(len(acc_gap)), acc_gap, alpha=0.3, color='green')
    
    # 4. Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    📊 TRAINING SUMMARY
    {'='*40}
    
    Final Metrics:
    • Train Accuracy: {history.history['accuracy'][-1]:.4f}
    • Val Accuracy:   {history.history['val_accuracy'][-1]:.4f}
    • Train Loss:     {history.history['loss'][-1]:.4f}
    • Val Loss:       {history.history['val_loss'][-1]:.4f}
    
    Best Metrics:
    • Best Train Acc: {max_train:.4f}
    • Best Val Acc:   {max_val:.4f}
    • Best Train Loss: {min_train:.4f}
    • Best Val Loss:   {min_val:.4f}
    
    Training Info:
    • Total Epochs: {len(history.history['accuracy'])}
    • Overfit Gap:  {acc_gap[-1]:.4f}
    
    Learning Rate:
    • Initial: 0.001
    """
    
    if 'lr' in history.history:
        summary_text += f"    • Final: {history.history['lr'][-1]:.6f}\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    # Save
    plot_path = os.path.join(save_path, f'{model_name}_training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {plot_path}")
    
    plt.show()


def print_training_summary(history, model_name, test_acc, training_time):
    """Print detailed training summary"""
    print("\n" + "="*60)
    print(f"📈 {model_name.upper()} - TRAINING SUMMARY")
    print("="*60)
    
    print("\n🎯 FINAL PERFORMANCE:")
    print(f"  Train Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"  Val Accuracy:   {history.history['val_accuracy'][-1]*100:.2f}%")
    print(f"  Test Accuracy:  {test_acc*100:.2f}%")
    
    print("\n🏆 BEST PERFORMANCE:")
    print(f"  Best Train Acc: {max(history.history['accuracy'])*100:.2f}%")
    print(f"  Best Val Acc:   {max(history.history['val_accuracy'])*100:.2f}%")
    
    print("\n📉 LOSS:")
    print(f"  Final Train Loss: {history.history['loss'][-1]:.4f}")
    print(f"  Final Val Loss:   {history.history['val_loss'][-1]:.4f}")
    print(f"  Best Train Loss:  {min(history.history['loss']):.4f}")
    print(f"  Best Val Loss:    {min(history.history['val_loss']):.4f}")
    
    print("\n⚠️ OVERFITTING CHECK:")
    acc_gap = history.history['accuracy'][-1] - history.history['val_accuracy'][-1]
    print(f"  Accuracy Gap: {acc_gap*100:.2f}%")
    if acc_gap > 0.05:
        print("  ⚠️ WARNING: Model may be overfitting!")
    else:
        print("  ✅ Good generalization")
    
    print(f"\n⏱️ TRAINING TIME: {training_time:.2f} seconds")
    print(f"⏱️ TIME PER EPOCH: {training_time/len(history.history['accuracy']):.2f} seconds")
    
    print("\n" + "="*60)


# ============================================================
# TRAINING
# ============================================================

def train_model(model_name, epochs=15, batch_size=128):
    """Train specified model with visualization"""
    
    print("="*60)
    print(f"EMNIST DIGIT RECOGNITION - {model_name.upper()} MODEL")
    print("="*60)
    
    # Load data
    with tf.device('/CPU:0'):
        X_train, y_train, X_test, y_test = load_emnist_digits()
    
    # Build model
    print(f"\n🔨 Building {model_name.upper()} Model...")
    if model_name.lower() == 'baseline':
        model = build_baseline_model()
    elif model_name.lower() == 'resnet':
        model = build_resnet_deep_model()
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'baseline' or 'resnet'")
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train
    print("\n🚀 Training...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Evaluate
    print("\n📈 Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    # Print summary
    print_training_summary(history, model_name, test_acc, training_time)
    
    # Visualize
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
    print(f"TensorFlow Version: {tf.__version__}")
    
    MODEL_NAME = 'resnet'
    
    train_model(
        model_name=MODEL_NAME,
        epochs=20,
        batch_size=128
    )