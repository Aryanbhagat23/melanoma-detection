# ============================================================
# Melanoma Detection - Two-Stage Fine-Tuning of ResNet50
# Author: Aryan Bhagat | FAU Computer Science
# Dataset: HAM10000 / ISIC 2025 version
# ============================================================
# FIX: Split FIRST, then oversample ONLY the training set.
# This prevents duplicate images from leaking between
# train and test sets, ensuring valid evaluation metrics.
# ============================================================

import os
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
)
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, accuracy_score,
    precision_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIG — Update these paths if needed
# ============================================================
IMAGES_ZIP  = "ISIC-images.zip"
IMAGES_DIR  = "ISIC-images"
LABELS_CSV  = "ham10000_metadata_2025-08-18.csv"
MODEL_SAVE  = "best_melanoma_model.keras"
BATCH_SIZE  = 32
EPOCHS_S1   = 15
EPOCHS_S2   = 30
LR_S1       = 1e-3
LR_S2       = 1e-5
IMG_SIZE    = (224, 224)
SEED        = 42

# ============================================================
# STEP 1 — Extract ZIP if not already extracted
# ============================================================
if not os.path.exists(IMAGES_DIR):
    print(f"Extracting {IMAGES_ZIP}...")
    with zipfile.ZipFile(IMAGES_ZIP, 'r') as z:
        z.extractall(".")
    print("Extraction complete!")
else:
    print(f"Images folder '{IMAGES_DIR}' already exists — skipping extraction.")

# ============================================================
# STEP 2 — Load metadata & build filepaths
# ============================================================
print("\nLoading metadata...")
df = pd.read_csv(LABELS_CSV)
print(f"Columns found: {list(df.columns)}")
print(f"Total rows: {len(df)}")

# Binary label: 1 = Malignant, 0 = Benign
df['label'] = df['diagnosis_1'].apply(
    lambda x: 1 if str(x).strip().lower() == 'malignant' else 0
)

print(f"\nClass distribution (full dataset):")
print(df['label'].value_counts())

def find_image(isic_id):
    for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
        path = os.path.join(IMAGES_DIR, isic_id + ext)
        if os.path.exists(path):
            return path
    return None

df['filepath'] = df['isic_id'].apply(find_image)
df = df[df['filepath'].notna()].reset_index(drop=True)
print(f"\nImages matched to metadata: {len(df)}")
print(f"Malignant: {df['label'].sum()} | Benign: {(df['label']==0).sum()}")

# ============================================================
# STEP 3 — SPLIT FIRST (before any oversampling)
# ============================================================
# This is the critical fix. We split on the ORIGINAL unbalanced
# dataset so no duplicate rows can appear in both train and test.
# ============================================================
print("\n--- Splitting dataset BEFORE oversampling ---")

X = df['filepath'].values
y = df['label'].values

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.20,
    stratify=y_trainval, random_state=SEED)

print(f"Train (before oversampling): {len(X_train)} "
      f"| Val: {len(X_val)} | Test: {len(X_test)}")
print(f"Test set — Malignant: {y_test.sum()} | Benign: {(y_test==0).sum()}")

# ============================================================
# STEP 4 — Oversample ONLY the TRAINING SET
# ============================================================
# Val and test sets are left untouched — real, unduped images only.
# ============================================================
print("\n--- Oversampling training set only ---")

train_df = pd.DataFrame({'filepath': X_train, 'label': y_train})
df_benign_tr   = train_df[train_df['label'] == 0]
df_melanoma_tr = train_df[train_df['label'] == 1]

print(f"Before: Benign={len(df_benign_tr)} | Malignant={len(df_melanoma_tr)}")

if len(df_melanoma_tr) < len(df_benign_tr):
    df_melanoma_up = resample(df_melanoma_tr, replace=True,
                              n_samples=len(df_benign_tr), random_state=SEED)
    train_balanced = pd.concat([df_benign_tr, df_melanoma_up])
else:
    df_benign_up = resample(df_benign_tr, replace=True,
                            n_samples=len(df_melanoma_tr), random_state=SEED)
    train_balanced = pd.concat([df_melanoma_tr, df_benign_up])

train_balanced = train_balanced.sample(
    frac=1, random_state=SEED).reset_index(drop=True)

X_train = train_balanced['filepath'].values
y_train = train_balanced['label'].values

print(f"After:  Total={len(X_train)} "
      f"| Malignant={y_train.sum()} | Benign={(y_train==0).sum()}")
print(f"\nFinal split — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ============================================================
# STEP 5 — tf.data Pipeline
# ============================================================
def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(tf.cast(img, tf.float32))
    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_saturation(img, 0.8, 1.2)
    return img, label

def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels.astype(np.float32)))
    ds = ds.map(load_and_preprocess,
                num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=2048, seed=SEED)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(X_train, y_train, training=True)
val_ds   = make_dataset(X_val,   y_val)
test_ds  = make_dataset(X_test,  y_test)

# ============================================================
# STEP 6 — Model Architecture
# ============================================================
print("\nBuilding model...")
base = ResNet50(weights='imagenet', include_top=False,
                input_shape=(*IMG_SIZE, 3))
x   = base.output
x   = GlobalAveragePooling2D()(x)
x   = Dense(1024, activation='relu')(x)
out = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base.input, outputs=out)
print(f"Total parameters: {model.count_params():,}")

# ============================================================
# STEP 7 — Stage 1: Train head only (base frozen)
# ============================================================
print("\n--- Stage 1: Training classification head (base frozen) ---")
for layer in base.layers:
    layer.trainable = False

model.compile(optimizer=Adam(LR_S1),
              loss='binary_crossentropy',
              metrics=['accuracy'])

cb_s1 = [ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                            patience=3, min_lr=1e-7, verbose=1)]
history_s1 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS_S1, callbacks=cb_s1, verbose=1)

# ============================================================
# STEP 8 — Stage 2: Full fine-tuning (all layers unfrozen)
# ============================================================
print("\n--- Stage 2: Global fine-tuning all layers (lr=1e-5) ---")
for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=Adam(LR_S2),
              loss='binary_crossentropy',
              metrics=['accuracy'])

cb_s2 = [
    EarlyStopping(monitor='val_loss', patience=10,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_SAVE, monitor='val_loss',
                    save_best_only=True, verbose=1)
]
history_s2 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS_S2, callbacks=cb_s2, verbose=1)

# ============================================================
# STEP 9 — Final Evaluation on Test Set
# ============================================================
print("\n--- Final Evaluation on Held-Out Test Set ---")

y_prob = model.predict(test_ds, verbose=1).ravel()
y_pred = (y_prob >= 0.5).astype(int)

cm           = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
acc          = accuracy_score(y_test, y_pred)
sens         = tp / (tp + fn)
spec         = tn / (tn + fp)
prec         = precision_score(y_test, y_pred)
f1           = f1_score(y_test, y_pred)
auc          = roc_auc_score(y_test, y_prob)

print("=" * 50)
print("Final Performance Metrics")
print("=" * 50)
print(f"Test set size:         {len(y_test)}")
print(f"  Malignant (actual):  {y_test.sum()}")
print(f"  Benign (actual):     {(y_test==0).sum()}")
print("-" * 50)
print(f"AUC-ROC:               {auc:.4f}")
print(f"Accuracy:              {acc:.4f}  ({acc*100:.2f}%)")
print(f"Sensitivity (Recall):  {sens:.4f}  ({sens*100:.2f}%)")
print(f"Specificity:           {spec:.4f}  ({spec*100:.2f}%)")
print(f"Precision:             {prec:.4f}  ({prec*100:.2f}%)")
print(f"F1-Score:              {f1:.4f}  ({f1*100:.2f}%)")
print("-" * 50)
print(f"TP={tp} | TN={tn} | FP={fp} | FN={fn}")
print("=" * 50)
print("\nNOTE: Test set contains only original images (no duplicates).")
print("Oversampling was applied to training set only.")

# ============================================================
# STEP 10 — Save Confusion Matrix Figure
# ============================================================
labels_text = [
    [f'TN = {tn}\n(Correct Benign)', f'FP = {fp}\n(False Alarm)'],
    [f'FN = {fn}\n(Missed Cancer)', f'TP = {tp}\n(Correct Melanoma)']
]
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=False, cmap='Blues', linewidths=2,
            linecolor='white', ax=ax,
            xticklabels=['Predicted Benign', 'Predicted Melanoma'],
            yticklabels=['Actual Benign', 'Actual Melanoma'])
for i in range(2):
    for j in range(2):
        ax.text(j+0.5, i+0.5, labels_text[i][j],
                ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')
ax.set_title(f'Confusion Matrix — Melanoma Detection\n'
             f'(Test Set N={len(y_test)})', fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=11)
ax.set_ylabel('Actual Label', fontsize=11)
fig.text(0.5, 0.01,
         f'Sensitivity: {sens:.4f}  |  Specificity: {spec:.4f}  |  '
         f'Accuracy: {acc:.4f}  |  AUC-ROC: {auc:.4f}',
         ha='center', fontsize=9, style='italic')
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('confusion_matrix.png', dpi=180, bbox_inches='tight')
print("\nConfusion matrix saved as 'confusion_matrix.png'")
print(f"Model saved as '{MODEL_SAVE}'")
print("\nDone! Run the detector app with: streamlit run melanoma_detector_app.py")