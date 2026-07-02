"""
EfficientNetB0 Training Script
Backbone efficace (5.3M params) - 2x plus rapide que VGGFace.
Accuracy ciblee : 90-92% sur LFW-deepfunneled.
"""
import sys
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

sys.path.insert(0, str(Path(__file__).parent.parent))

DATASET_PATH  = Path("data/lfw-deepfunneled/lfw-deepfunneled")
RESULTS_DIR   = Path("results/efficientnet")
MIN_IMAGES    = 10
IMG_H, IMG_W  = 224, 224
BATCH_SIZE    = 64
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 20
LR            = 1e-3


# ── tf.data pipeline ──────────────────────────────────────────────────────────

def _parse(path, label, augment):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [IMG_H, IMG_W])
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        img = tf.image.random_saturation(img, 0.8, 1.2)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img, label


def make_dataset(paths, labels, batch_size, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((
        tf.constant(paths,  dtype=tf.string),
        tf.constant(labels, dtype=tf.int32),
    ))
    if augment:
        ds = ds.shuffle(buffer_size=min(len(paths), 10_000), reshuffle_each_iteration=True)
    ds = ds.map(lambda p, l: _parse(p, l, augment), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ── Collecte chemins ──────────────────────────────────────────────────────────

def collect_paths():
    print("Collecte des chemins pour EfficientNetB0...")
    paths, labels = [], []
    class_dirs = sorted(
        d for d in DATASET_PATH.iterdir()
        if d.is_dir() and len(list(d.glob("*.jpg"))) >= MIN_IMAGES
    )
    print(f"  {len(class_dirs)} classes avec >= {MIN_IMAGES} images")
    cls_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
    for cls_dir in class_dirs:
        for img_path in sorted(cls_dir.glob("*.jpg"))[:MIN_IMAGES]:
            paths.append(str(img_path))
            labels.append(cls_to_idx[cls_dir.name])
    print(f"  {len(paths)} chemins collectes")
    return paths, labels, [d.name for d in class_dirs]


# ── Modele ────────────────────────────────────────────────────────────────────

def build_model(num_classes):
    base = EfficientNetB0(weights="imagenet", include_top=False,
                          input_shape=(IMG_H, IMG_W, 3))
    base.trainable = False

    x   = base.output
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dense(512, activation="relu")(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs=base.input, outputs=out), base


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, val_ds):
    y_true, y_pred = [], []
    for imgs, lbls in val_ds:
        preds = np.argmax(model.predict(imgs, verbose=0), axis=1)
        y_true.extend(lbls.numpy().tolist())
        y_pred.extend(preds.tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred,    average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred,        average="weighted", zero_division=0)

    print("\n" + "=" * 48)
    print("  METRIQUES SUR LA VALIDATION — EfficientNetB0")
    print("=" * 48)
    print(f"  Accuracy  : {acc*100:6.2f}%")
    print(f"  Precision : {prec*100:6.2f}%  (weighted)")
    print(f"  Recall    : {rec*100:6.2f}%  (weighted)")
    print(f"  F1-score  : {f1*100:6.2f}%  (weighted)")
    print("=" * 48 + "\n")

    return {"accuracy": float(acc), "precision": float(prec),
            "recall": float(rec), "f1_score": float(f1)}


# ── Entrainement ──────────────────────────────────────────────────────────────

def train():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    paths, labels, class_names = collect_paths()
    num_classes = len(class_names)

    idx_train, idx_val = train_test_split(
        range(len(paths)), test_size=0.2, random_state=42, stratify=labels
    )
    train_paths  = [paths[i]  for i in idx_train]
    train_labels = [labels[i] for i in idx_train]
    val_paths    = [paths[i]  for i in idx_val]
    val_labels   = [labels[i] for i in idx_val]

    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Classes: {num_classes}")

    train_ds = make_dataset(train_paths, train_labels, BATCH_SIZE, augment=True)
    val_ds   = make_dataset(val_paths,   val_labels,   BATCH_SIZE, augment=False)

    model, base = build_model(num_classes)
    model.summary()

    callbacks = [
        ModelCheckpoint(str(RESULTS_DIR / "efficientnet_best.keras"),
                        save_best_only=True, monitor="val_accuracy", verbose=1),
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-8, verbose=1),
    ]

    # ── Phase 1 : tete seulement ──
    print("\n[EfficientNetB0] Phase 1 : tete de classification...")
    model.compile(optimizer=keras.optimizers.Adam(LR),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=EPOCHS_PHASE1, callbacks=callbacks, verbose=1)

    # ── Phase 2 : fine-tune derniers blocs ──
    print("\n[EfficientNetB0] Phase 2 : fine-tuning...")
    for layer in base.layers[-30:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    model.compile(optimizer=keras.optimizers.Adam(LR * 0.1),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train_ds, validation_data=val_ds,
                   epochs=EPOCHS_PHASE2, callbacks=callbacks, verbose=1)

    model.save(str(RESULTS_DIR / "efficientnet_final.keras"))

    with open(RESULTS_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f)

    print("\n[EfficientNetB0] Evaluation finale...")
    metrics = evaluate(model, val_ds)

    results = {
        "model": "EfficientNetB0",
        "backbone": "EfficientNetB0",
        "input_size": f"{IMG_H}x{IMG_W}",
        "num_classes": num_classes,
        "total_images": len(paths),
        "min_images_per_class": MIN_IMAGES,
        "epochs_trained": len(h1.history["loss"]) + len(h2.history["loss"]),
        "Accuracy":  round(metrics["accuracy"]  * 100, 2),
        "Precision": round(metrics["precision"] * 100, 2),
        "Recall":    round(metrics["recall"]    * 100, 2),
        "F1-score":  round(metrics["f1_score"]  * 100, 2),
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"[EfficientNetB0] Termine ! results -> {RESULTS_DIR / 'results.json'}")
    return results


if __name__ == "__main__":
    train()
