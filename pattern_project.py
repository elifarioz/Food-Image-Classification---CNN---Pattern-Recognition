# ================================================================
# AAI 646-A — FOOD IMAGE CLASSIFICATION
# ================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2

# ---------------------------------------------------------
# 1. DIRECTORY SETUP
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_dir = os.path.join(BASE_DIR, 'food11', 'training')
val_dir = os.path.join(BASE_DIR, 'food11', 'validation')
test_dir = os.path.join(BASE_DIR, 'food11', 'evaluation')

IMG_SIZE = 150
BATCH_SIZE = 32

# ---------------------------------------------------------
# 2. DATA PREPROCESSING + AUGMENTATION (IMPROVED)
# ---------------------------------------------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ---------------------------------------------------------
# 3. CLASS WEIGHTS (Class Imbalance)
# ---------------------------------------------------------


from collections import Counter

print("\n=== CALSS DISTRIBUTION ===")
class_counts = Counter(train_gen.classes)
for cls, count in sorted(class_counts.items()):
    class_name = list(train_gen.class_indices.keys())[cls]
    print(f"{class_name:20s}: {count:5d} samples")

# Class weights
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights_dict = dict(enumerate(class_weights))

print("\n=== CLASS WEIGHTS ===")
for idx, weight in class_weights_dict.items():
    cls_name = list(train_gen.class_indices.keys())[idx]
    print(f"{cls_name}: {weight:.2f}")

# ---------------------------------------------------------
# 4. IMPROVED CNN MODEL
# ---------------------------------------------------------

from keras.layers import Input

model = Sequential([
    # Input Layer
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same',
           kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same',
           kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same',
           kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same',
           kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same',
           kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same',
           kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Block 4
    Conv2D(256, (3, 3), activation='relu', padding='same',
           kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # Classifier
    GlobalAveragePooling2D(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.4),
    Dense(11, activation='softmax')
])
# learning rate with Adam optimizer
from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------------------------------------------------
# 5. TRAINING WITH CLASS WEIGHTS
# ---------------------------------------------------------

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

print("\n=== TRAINING STARTED ===")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    class_weight=class_weights_dict,  # CLASS WEIGHTS KULLAN!
    callbacks=[lr_scheduler, early_stop],
    verbose=1
)

# ---------------------------------------------------------
# 6. MODEL EVALUATION
# ---------------------------------------------------------

print("\n=== EVALUATING ON TEST SET ===")
test_gen.reset()
pred_probs = model.predict(test_gen, verbose=1)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(true_classes, pred_classes, target_names=class_labels))

# Overall accuracy
accuracy = np.mean(pred_classes == true_classes)
print(f"\n📊 Overall Test Accuracy: {accuracy*100:.2f}%")

# ---------------------------------------------------------
# 7. CONFUSION MATRIX
# ---------------------------------------------------------

cm = confusion_matrix(true_classes, pred_classes)

plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_labels,
            yticklabels=class_labels,
            cmap='RdYlGn',
            linewidths=0.5)
plt.xlabel("Predicted", fontsize=12, fontweight='bold')
plt.ylabel("True", fontsize=12, fontweight='bold')
plt.title("Confusion Matrix – Food-11 CNN (Improved)",
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('confusion_matrix_improved.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------
# 8. TRAINING HISTORY
# ---------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Model Accuracy Over Time', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Model Loss Over Time', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_improved.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------
# 9. PER-CLASS ACCURACY
# ---------------------------------------------------------

print("\n" + "="*60)
print("PER-CLASS ACCURACY")
print("="*60)

for i, class_name in enumerate(class_labels):
    class_mask = true_classes == i
    class_acc = np.mean(pred_classes[class_mask] == true_classes[class_mask])
    print(f"{class_name:20s}: {class_acc*100:6.2f}%")

print("\n✅ Training Complete! Model saved metrics to images.")