"""
VGGFace Training Script
VGG16 backbone (ImageNet pre-trained) fine-tune sur LFW.
Chargement paresseux via tf.data pour eviter l'OOM (32 GiB si tout en RAM).
Avec métriques détaillées et visualisations.
"""
import sys
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

sys.path.insert(0, str(Path(__file__).parent.parent))

DATASET_PATH  = Path("data/lfw-deepfunneled/lfw-deepfunneled")
RESULTS_DIR   = Path("results/vggface")
MIN_IMAGES    = 10
IMG_H, IMG_W  = 224, 224
BATCH_SIZE    = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 20
LR            = 1e-4


# ── tf.data pipeline (chargement par batch depuis le disque) ──────────────────

def _parse(path: tf.Tensor, label: tf.Tensor, augment: bool):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [IMG_H, IMG_W])
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img, label


def make_dataset(paths, labels, batch_size, augment=False):
    paths_t  = tf.constant(paths,  dtype=tf.string)
    labels_t = tf.constant(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths_t, labels_t))
    if augment:
        ds = ds.shuffle(buffer_size=min(len(paths), 10_000), reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, l: _parse(p, l, augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Collecte des chemins (sans charger les pixels) ────────────────────────────

def collect_paths():
    print("Collecte des chemins pour VGGFace...")
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

    class_names = [d.name for d in class_dirs]
    print(f"  {len(paths)} chemins collectes (0 octet RAM image)")
    return paths, labels, class_names


# ── Modele ────────────────────────────────────────────────────────────────────

def build_model(num_classes):
    base = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_H, IMG_W, 3))
    for layer in base.layers:
        layer.trainable = False

    x   = base.output
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(4096, activation="relu")(x)
    x   = layers.Dropout(0.5)(x)
    x   = layers.Dense(4096, activation="relu")(x)
    x   = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs=base.input, outputs=out), base


# ── Visualisations et métriques ──────────────────────────────────────────────

def plot_training_history(h1, h2, save_path):
    """Trace les courbes d'accuracy et de loss"""
    # Combiner les historiques
    train_acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    train_loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']
    
    epochs = range(1, len(train_acc) + 1)
    separator = len(h1.history['accuracy'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(epochs, train_acc, 'b-', label='Entraînement', linewidth=2)
    ax1.plot(epochs, val_acc, 'r-', label='Validation', linewidth=2)
    ax1.axvline(x=separator, color='gray', linestyle='--', alpha=0.7, label='Début Phase 2')
    ax1.set_xlabel('Époques')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Évolution de l\'accuracy - VGGFace')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(epochs, train_loss, 'b-', label='Entraînement', linewidth=2)
    ax2.plot(epochs, val_loss, 'r-', label='Validation', linewidth=2)
    ax2.axvline(x=separator, color='gray', linestyle='--', alpha=0.7, label='Début Phase 2')
    ax2.set_xlabel('Époques')
    ax2.set_ylabel('Loss')
    ax2.set_title('Évolution de la loss - VGGFace')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'vggface_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Graphique d'historique sauvegardé: {save_path / 'vggface_history.png'}")


def calculer_metriques_detaillees(model, dataset, class_names, save_path):
    """Calcule et affiche Accuracy, Precision, Recall, F1-score"""
    print("\n" + "=" * 60)
    print("📊 CALCUL DES MÉTRIQUES DÉTAILLÉES")
    print("=" * 60)
    
    # Récupérer les prédictions
    y_true = []
    y_pred = []
    
    print("🔄 Récupération des prédictions sur l'ensemble de validation...")
    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(y_batch.numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calcul des métriques
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # ============================================================
    # AFFICHAGE CONSOLE
    # ============================================================
    print("\n" + "=" * 60)
    print("🏆 RÉSULTATS SUR L'ENSEMBLE DE VALIDATION")
    print("=" * 60)
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-score:  {f1:.4f} ({f1*100:.2f}%)")
    print("=" * 60)
    
    # Appréciation
    print("\n📈 APPRÉCIATION:")
    if f1 >= 0.95:
        print("  ✅ EXCELLENT - Le modèle est parfaitement adapté")
    elif f1 >= 0.90:
        print("  👍 TRÈS BON - Résultat très satisfaisant")
    elif f1 >= 0.80:
        print("  ⚠️ CORRECT - Peut être amélioré avec plus de données")
    else:
        print("  ❌ INSUFFISANT - Revoir l'architecture ou les paramètres")
    print("=" * 60)
    
    # ============================================================
    # SAUVEGARDE JSON
    # ============================================================
    metriques = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'n_samples': len(y_true)
    }
    
    with open(save_path / 'metriques_detaillees.json', 'w') as f:
        json.dump(metriques, f, indent=2)
    print(f"\n✅ Métriques sauvegardées: {save_path / 'metriques_detaillees.json'}")
    
    # ============================================================
    # GRAPHIQUE À BARRES
    # ============================================================
    plt.figure(figsize=(10, 6))
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    metrics_values = [accuracy, precision, recall, f1]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.8)
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title(f'VGGFace - Métriques de performance\n(F1-score: {f1:.2%})')
    
    for bar, val in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2%}', ha='center', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path / 'vggface_metriques.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Graphique des métriques sauvegardé: {save_path / 'vggface_metriques.png'}")
    
    # ============================================================
    # RAPPORT DE CLASSIFICATION (10 premières classes)
    # ============================================================
    print("\n📋 RAPPORT DE CLASSIFICATION (10 premières classes):")
    print("-" * 70)
    
    unique_classes = np.unique(y_true)[:10]
    target_names = [class_names[i].replace('_', ' ') for i in unique_classes if i < len(class_names)]
    
    if len(target_names) > 0:
        mask = np.isin(y_true, unique_classes)
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        report = classification_report(y_true_filtered, y_pred_filtered, 
                                        target_names=target_names,
                                        zero_division=0)
        print(report)
    
    return metriques


def plot_confusion_matrix(model, dataset, class_names, save_path, num_classes=20):
    """Trace la matrice de confusion sur les num_classes premieres classes."""
    print("\n📊 Calcul de la matrice de confusion...")

    y_true, y_pred = [], []

    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(y_batch.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Garder uniquement les echantillons dont le label est dans [0, num_classes[
    mask   = (y_true < num_classes) & (y_pred < num_classes)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        print("⚠️ Pas assez d'echantillons pour tracer la matrice de confusion.")
        return

    labels     = list(range(num_classes))
    tick_names = [class_names[i].replace('_', ' ') for i in labels]
    cm         = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=tick_names, yticklabels=tick_names)
    plt.title(f'VGGFace - Matrice de confusion (premieres {num_classes} classes)')
    plt.xlabel('Prediction')
    plt.ylabel('Reel')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path / 'vggface_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Matrice de confusion sauvegardee: {save_path / 'vggface_confusion_matrix.png'}")


# ── Entrainement ──────────────────────────────────────────────────────────────

def train():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    paths, labels, class_names = collect_paths()
    num_classes = len(class_names)

    idx = list(range(len(paths)))
    idx_train, idx_val = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=labels
    )

    train_paths  = [paths[i]  for i in idx_train]
    train_labels = [labels[i] for i in idx_train]
    val_paths    = [paths[i]  for i in idx_val]
    val_labels   = [labels[i] for i in idx_val]

    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Classes: {num_classes}")

    train_ds = make_dataset(train_paths, train_labels, BATCH_SIZE, augment=True)
    val_ds   = make_dataset(val_paths,   val_labels,   BATCH_SIZE, augment=False)

    model, base = build_model(num_classes)

    callbacks = [
        ModelCheckpoint(
            str(RESULTS_DIR / "vggface_best.keras"),
            save_best_only=True, monitor="val_accuracy", verbose=1,
        ),
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-8, verbose=1),
    ]

    # ── Phase 1 : tete seulement ──
    print("\n" + "=" * 60)
    print("[VGGFace] Phase 1 : entrainement de la tete de classification...")
    print("=" * 60)
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE1,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Phase 2 : fine-tune dernier bloc conv VGG16 ──
    print("\n" + "=" * 60)
    print("[VGGFace] Phase 2 : fine-tuning bloc conv5 de VGG16...")
    print("=" * 60)
    for layer in base.layers[-4:]:
        layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(LR * 0.1),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE2,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(str(RESULTS_DIR / "vggface_final.keras"))

    with open(RESULTS_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f)

    best_acc = max(
        h1.history.get("val_accuracy", [0]) + h2.history.get("val_accuracy", [0])
    )
    
    # ============================================================
    # GÉNÉRATION DES VISUALISATIONS ET MÉTRIQUES
    # ============================================================
    print("\n" + "=" * 60)
    print("📊 GÉNÉRATION DES VISUALISATIONS")
    print("=" * 60)
    
    # 1. Courbes d'entraînement
    plot_training_history(h1, h2, RESULTS_DIR)
    
    # 2. Métriques détaillées sur l'ensemble de validation
    metriques = calculer_metriques_detaillees(model, val_ds, class_names, RESULTS_DIR)
    
    # 3. Matrice de confusion (optionnelle - peut être lente)
    plot_confusion_matrix(model, val_ds, class_names, RESULTS_DIR, num_classes=20)
    
    # ============================================================
    # RÉSULTATS FINAUX
    # ============================================================
    results = {
        "model": "VGGFace",
        "backbone": "VGG16",
        "num_classes": num_classes,
        "total_images": len(paths),
        "min_images_per_class": MIN_IMAGES,
        "best_val_accuracy": float(best_acc),
        "epochs_trained": len(h1.history["loss"]) + len(h2.history["loss"]),
        "precision": metriques['precision'],
        "recall": metriques['recall'],
        "f1_score": metriques['f1_score']
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ============================================================
    # AFFICHAGE FINAL
    # ============================================================
    print("\n" + "=" * 60)
    print(" ENTRAÎNEMENT TERMINÉ !")
    print("=" * 60)
    print(f"  Meilleure val_accuracy : {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"  Precision (weighted)   : {metriques['precision']:.4f} ({metriques['precision']*100:.2f}%)")
    print(f"  Recall (weighted)      : {metriques['recall']:.4f} ({metriques['recall']*100:.2f}%)")
    print(f"  F1-score (weighted)    : {metriques['f1_score']:.4f} ({metriques['f1_score']*100:.2f}%)")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    train()