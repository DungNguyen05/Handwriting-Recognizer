"""
CNN ARCHITECTURE EXPERIMENTS
So sánh chi tiết các kiến trúc CNN khác nhau cho EMNIST
"""

import emnist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import string
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import time
import json

# Setup
np.random.seed(42)
tf.random.set_seed(42)
os.makedirs('experiments', exist_ok=True)
os.makedirs('experiments/models', exist_ok=True)
digits = list(string.digits)  # 0-9
uppercase = list(string.ascii_uppercase)  # A-Z
lowercase = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
labels_list = uppercase + lowercase # Total: 37 classes (No digits)

print("="*70)
print("CNN ARCHITECTURE EXPERIMENTS")
print("="*70)
print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

# ==================== LOAD DATA ====================
print("\n📦 Loading EMNIST Letters...")
X_train_raw, y_train_raw = emnist.extract_training_samples('balanced')
X_test_raw, y_test_raw = emnist.extract_test_samples('balanced')
    
# Filter out digits (0-9)
# In EMNIST Balanced: 0-9 are digits, 10-46 are letters
train_mask = y_train_raw >= 10
test_mask = y_test_raw >= 10

X_train_raw = X_train_raw[train_mask]
y_train_raw = y_train_raw[train_mask] - 10 # Shift to 0-36

X_test_raw = X_test_raw[test_mask]
y_test_raw = y_test_raw[test_mask] - 10 # Shift to 0-36

# Preprocess
X_train = X_train_raw.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test_raw.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train_raw, 37)
y_test = to_categorical(y_test_raw, 37)

print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
print(f"✅ Classes: 37 (Letters Only)")
print(f"✅ Label range: {y_train_raw.min()} to {y_train_raw.max()}")

# ==================== ARCHITECTURE DEFINITIONS ====================
print("\n" + "="*70)
print("DEFINING ARCHITECTURES")
print("="*70)

# Data Augmentation (Rotation ~30deg, Zoom ~10%)
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.08),
])

def architecture_1_baseline():
    """
    BASELINE - Simple CNN (Your original)
    - 2 Conv blocks
    - Basic structure
    - ~400K parameters
    """
    model = models.Sequential([
        # data_augmentation,
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        # layers.Dropout(0.2),
        layers.Dense(37, activation='softmax')
    ], name='Baseline')
    return model

def architecture_2_deeper():
    """
    DEEPER CNN - More conv blocks
    - 4 Conv blocks
    - More capacity
    - ~400K parameters
    """
    model = models.Sequential([
        # data_augmentation,
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        # layers.Dropout(0.5),
        layers.Dense(37, activation='softmax')
    ], name='Deeper')
    return model

def architecture_3_wider():
    """
    WIDER CNN - More filters per layer
    - 2 Conv blocks but WIDER (128, 256 filters)
    - More parameters
    - ~1M parameters
    """
    model = models.Sequential([
        # data_augmentation,
        layers.Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        # layers.Dropout(0.5),
        layers.Dense(37, activation='softmax')
    ], name='Wider')
    return model

def architecture_4_resnet_style():
    """
    RESNET-STYLE - With residual connections
    - Skip connections
    - Deeper but stable training
    - ~300K parameters
    """
    inputs = layers.Input(shape=(28, 28, 1))
    
    # Augmentation
    # x = data_augmentation(inputs)
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
    
    # Dense
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    # x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(37, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name='ResNet_Style')
    return model

def architecture_7_resnet_deep():
    """
    RESNET-DEEPER - More residual blocks
    - 3 Residual blocks with increasing filters
    - Deeper feature extraction
    """
    inputs = layers.Input(shape=(28, 28, 1))
    
    # Augmentation
    # x = data_augmentation(inputs)
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
    # x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(37, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name='ResNet_Deep')
    return model

def architecture_5_large_kernels():
    """
    LARGE KERNELS - Using 5x5 and 7x7 kernels
    - Larger receptive field
    - Fewer layers needed
    - ~500K parameters
    """
    model = models.Sequential([
        # data_augmentation,
        layers.Conv2D(32, (7, 7), padding='same', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (5, 5), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        # layers.Dropout(0.5),
        layers.Dense(37, activation='softmax')
    ], name='Large_Kernels')
    return model

def architecture_6_no_batchnorm():
    """
    NO BATCH NORM - Test impact of BatchNorm
    - Same as baseline but without BN
    - Compare performance
    """
    model = models.Sequential([
        # data_augmentation,
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        # layers.Dropout(0.4),
        layers.Dense(37, activation='softmax')
    ], name='No_BatchNorm')
    return model

# Dictionary of all architectures
ARCHITECTURES = {
    'Baseline': architecture_1_baseline,
    'Deeper': architecture_2_deeper,
    'Wider': architecture_3_wider,
    'ResNet_Style': architecture_4_resnet_style,
    'ResNet_Deep': architecture_7_resnet_deep,
    'Large_Kernels': architecture_5_large_kernels,
    'No_BatchNorm': architecture_6_no_batchnorm,
}

print(f"\n📐 Defined {len(ARCHITECTURES)} architectures:")
for name in ARCHITECTURES.keys():
    print(f"  - {name}")

# ==================== TRAINING FUNCTION ====================

def visualize_performance(history, model, X_test, y_test, name):
    print(f"\n📊 Creating visualizations for {name}...")
    os.makedirs('experiments', exist_ok=True)
    
    # 1. Training Curves (Loss & Accuracy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2, linestyle='--')
    ax1.set_title(f'{name} - Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy
    ax2.plot(history.history['accuracy'], label='Train Acc', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Val Acc', linewidth=2, linestyle='--')
    ax2.set_title(f'{name} - Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'experiments/{name}_training_curves.png', dpi=150)
    plt.close()
    
    # 2. Test Samples (Predictions)
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    indices = np.random.choice(len(X_test), 25, replace=False)
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    fig.suptitle(f'{name} - Test Samples', fontsize=16)
    
    for i, idx in enumerate(indices):
        ax = axes[i//5, i%5]
        img = X_test[idx].reshape(28, 28)
        true_label = labels_list[y_true[idx]]
        pred_label = labels_list[y_pred[idx]]
        
        ax.imshow(img, cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color, fontweight='bold')
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(f'experiments/{name}_test_samples.png', dpi=150)
    plt.close()

    # 3. Error Histogram
    errors = y_true[y_true != y_pred]
    if len(errors) > 0:
        plt.figure(figsize=(12, 6))
        sns.histplot(errors, bins=range(38), kde=False, color='salmon')
        plt.xticks(range(37), labels_list)
        plt.title(f'{name} - Error Distribution by True Label', fontsize=14, fontweight='bold')
        plt.xlabel('True Label')
        plt.ylabel('Count of Misclassifications')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(f'experiments/{name}_error_histogram.png', dpi=150)
        plt.close()

def train_and_evaluate(model, name, epochs=15, batch_size=128):
    """
    Train and evaluate a model
    Returns: metrics dictionary
    """
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print('='*60)
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Model summary
    model.summary()

    # Visualize the model
    keras.utils.plot_model(model, to_file=f'data/model_{name}.png', show_shapes=True, show_layer_names=True)
    print(f"\n✅ Đã lưu: data/model_{name}.png") 
    
    # Summary
    print(f"\n📊 Model Summary:")
    if not model.built:
        model.build((None, 28, 28, 1))
    params = model.count_params()
    print(f"  Total parameters: {params:,}")
    
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
            min_lr=1e-7,
            verbose=1
        ),
    ]
    
    # Train
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    train_time = time.time() - start_time
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    
    # Visualize Performance (New)
    visualize_performance(history, model, X_test, y_test, name)
    
    # Predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate per-class accuracy
    class_accs = []
    for i in range(37):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).sum() / mask.sum()
            class_accs.append(acc)
    avg_class_acc = np.mean(class_accs)
    
    # Results
    results = {
        'name': name,
        'params': params,
        'train_time': train_time,
        'epochs_trained': len(history.history['loss']),
        'final_train_acc': history.history['accuracy'][-1],
        'final_val_acc': history.history['val_accuracy'][-1],
        'test_acc': test_acc,
        'test_loss': test_loss,
        'avg_class_acc': avg_class_acc,
        'history': history.history,
        'predictions': (y_true, y_pred)
    }
    
    print(f"\n✅ Results:")
    print(f"  - Test Accuracy: {test_acc*100:.2f}%")
    print(f"  - Avg Class Accuracy: {avg_class_acc*100:.2f}%")
    print(f"  - Training Time: {train_time:.1f}s")
    print(f"  - Epochs: {results['epochs_trained']}")
    
    # Save model
    model.save(f"experiments/models/{name}.h5")
    
    return results

# ==================== RUN EXPERIMENTS ====================
print("\n" + "="*70)
print("RUNNING EXPERIMENTS")
print("="*70)

all_results = []

for name, arch_fn in ARCHITECTURES.items():
    print(f"\n🔬 Experiment: {name}")
    
    # Build model
    model = arch_fn()
    
    # Train and evaluate
    results = train_and_evaluate(model, name, epochs=15, batch_size=128)
    all_results.append(results)
    
    # Clear session to free memory
    keras.backend.clear_session()

print("\n✅ All experiments completed!")

# ==================== SAVE RESULTS ====================
print("\n💾 Saving results...")

# Save as JSON (without history and predictions)
results_summary = []
for r in all_results:
    summary = {k: v for k, v in r.items() if k not in ['history', 'predictions']}
    results_summary.append(summary)

with open('experiments/results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("✅ Saved: experiments/results_summary.json")

# ==================== COMPARISON ANALYSIS ====================
print("\n" + "="*70)
print("COMPARISON ANALYSIS")
print("="*70)

# Create comparison dataframe
df = pd.DataFrame([{
    'Architecture': r['name'],
    'Params (K)': r['params'] / 1000,
    'Test Acc (%)': r['test_acc'] * 100,
    'Avg Class Acc (%)': r['avg_class_acc'] * 100,
    'Train Time (s)': r['train_time'],
    'Epochs': r['epochs_trained']
} for r in all_results])

df = df.sort_values('Test Acc (%)', ascending=False)

print("\n📊 Results Table:")
print(df.to_string(index=False))

# Save to CSV
df.to_csv('experiments/comparison.csv', index=False)
print("\n✅ Saved: experiments/comparison.csv")

# ==================== VISUALIZATIONS ====================
print("\n📈 Creating visualizations...")

# 1. Accuracy comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Test Accuracy
ax = axes[0, 0]
bars = ax.barh(df['Architecture'], df['Test Acc (%)'], color='steelblue')
ax.set_xlabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
           f'{width:.2f}%', va='center', fontsize=10)

# Parameters vs Accuracy
ax = axes[0, 1]
ax.scatter(df['Params (K)'], df['Test Acc (%)'], s=200, alpha=0.6, c='coral')
for i, row in df.iterrows():
    ax.annotate(row['Architecture'], 
               (row['Params (K)'], row['Test Acc (%)']),
               fontsize=9, ha='center')
ax.set_xlabel('Parameters (K)', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Parameters vs Accuracy', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

# Training Time
ax = axes[1, 0]
bars = ax.barh(df['Architecture'], df['Train Time (s)'], color='lightgreen')
ax.set_xlabel('Training Time (seconds)', fontsize=12)
ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + 5, bar.get_y() + bar.get_height()/2,
           f'{width:.0f}s', va='center', fontsize=10)

# Efficiency (Accuracy / Time)
df['Efficiency'] = df['Test Acc (%)'] / df['Train Time (s)']
ax = axes[1, 1]
bars = ax.barh(df['Architecture'], df['Efficiency'], color='plum')
ax.set_xlabel('Efficiency (Acc% / Second)', fontsize=12)
ax.set_title('Training Efficiency', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
           f'{width:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('experiments/comparison_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Saved: experiments/comparison_analysis.png")

# 2. Training curves comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for r in all_results:
    axes[0].plot(r['history']['val_accuracy'], label=r['name'], alpha=0.7, linewidth=2)
    axes[1].plot(r['history']['val_loss'], label=r['name'], alpha=0.7, linewidth=2)

axes[0].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend(fontsize=9, loc='lower right')
axes[0].grid(alpha=0.3)

axes[1].set_title('Validation Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend(fontsize=9, loc='upper right')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('experiments/training_curves.png', dpi=150, bbox_inches='tight')
print("✅ Saved: experiments/training_curves.png")

# 3. Confusion matrices for top 3 models
top_3 = df.head(3)['Architecture'].values

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Confusion Matrices - Top 3 Models', fontsize=14, fontweight='bold')

for idx, arch_name in enumerate(top_3):
    result = next(r for r in all_results if r['name'] == arch_name)
    y_true, y_pred = result['predictions']
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm, annot=False, cmap='Blues', ax=axes[idx],
               xticklabels=labels_list, yticklabels=labels_list)
    axes[idx].set_title(f'{arch_name}\nAcc: {result["test_acc"]*100:.2f}%', 
                       fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('True')

plt.tight_layout()
plt.savefig('experiments/confusion_matrices_top3.png', dpi=150, bbox_inches='tight')
print("✅ Saved: experiments/confusion_matrices_top3.png")

# ==================== INSIGHTS & RECOMMENDATIONS ====================
print("\n" + "="*70)
print("INSIGHTS & RECOMMENDATIONS")
print("="*70)

best_acc = df.iloc[0]
best_efficient = df.sort_values('Efficiency', ascending=False).iloc[0]
fastest = df.sort_values('Train Time (s)').iloc[0]
smallest = df.sort_values('Params (K)').iloc[0]

print(f"""
🏆 BEST OVERALL ACCURACY:
  - Architecture: {best_acc['Architecture']}
  - Test Accuracy: {best_acc['Test Acc (%)']:.2f}%
  - Parameters: {best_acc['Params (K)']:.0f}K
  - Training Time: {best_acc['Train Time (s)']:.0f}s

⚡ MOST EFFICIENT (Acc/Time):
  - Architecture: {best_efficient['Architecture']}
  - Efficiency: {best_efficient['Efficiency']:.3f}
  - Test Accuracy: {best_efficient['Test Acc (%)']:.2f}%

🚀 FASTEST TRAINING:
  - Architecture: {fastest['Architecture']}
  - Training Time: {fastest['Train Time (s)']:.0f}s
  - Test Accuracy: {fastest['Test Acc (%)']:.2f}%

💾 SMALLEST MODEL:
  - Architecture: {smallest['Architecture']}
  - Parameters: {smallest['Params (K)']:.0f}K
  - Test Accuracy: {smallest['Test Acc (%)']:.2f}%

📊 KEY FINDINGS:

1. Batch Normalization Impact:
   - Compare '{[r['name'] for r in all_results if 'BatchNorm' not in r['name']][0]}' vs 'No_BatchNorm'
   - BatchNorm typically adds ~2-5% accuracy
   - Stabilizes training significantly

2. Architecture Recommendations:
   - For BEST accuracy: {best_acc['Architecture']}
   - For DEPLOYMENT (size matters): {smallest['Architecture']}
   - For RESEARCH (balanced): {best_efficient['Architecture']}
""")

# ==================== SUMMARY ====================
print("\n" + "="*70)
print("EXPERIMENT SUMMARY")
print("="*70)

print(f"""
✅ COMPLETED {len(ARCHITECTURES)} EXPERIMENTS

📁 Output Files:
  - experiments/results_summary.json
  - experiments/comparison.csv
  - experiments/comparison_analysis.png
  - experiments/training_curves.png
  - experiments/confusion_matrices_top3.png
  - experiments/models/*.h5 (7 trained models)
""")

print("="*70)
print("🎉 EXPERIMENTS COMPLETED!")
print("="*70)