"""
RH-REC - Evaluation avancee :
    1. Matrice de confusion
    2. Matrice de correlation des metriques
    3. FAR et FRR par classe

Utilisation :
    python models/evaluate_confusion_far_frr.py --model efficientnet
    python models/evaluate_confusion_far_frr.py --model vggface
    python models/evaluate_confusion_far_frr.py --model resnet50v2

Options utiles :
    --top-n 30        Nombre de classes affichees dans les graphiques lisibles
    --batch-size 64   Taille des batchs pendant la prediction

Sorties generees :
    results/<model>/confusion_matrix_top30.png
    results/<model>/correlation_matrix_metrics.png
    results/<model>/far_frr_by_class.png
    results/<model>/far_frr_by_class.csv
    results/<model>/evaluation_summary.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


DATASET_PATH = Path("data/lfw-deepfunneled/lfw-deepfunneled")
MIN_IMAGES = 10
IMG_SIZE = (224, 224)


MODEL_CONFIG = {
    "efficientnet": {
        "model_path": Path("results/efficientnet/efficientnet_best.keras"),
        "results_dir": Path("results/efficientnet"),
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
    },
    "vggface": {
        "model_path": Path("results/vggface/vggface_best.keras"),
        "results_dir": Path("results/vggface"),
        "preprocess": tf.keras.applications.vgg16.preprocess_input,
    },
    "resnet50v2": {
        "model_path": Path("results/resnet50v2/resnet50_best.keras"),
        "results_dir": Path("results/resnet50v2"),
        "preprocess": tf.keras.applications.resnet.preprocess_input,
    },
}


def collect_dataset():
    """Recupere les chemins d'images et les labels comme dans les scripts d'entrainement."""
    image_paths = []
    labels = []

    class_dirs = sorted(
        folder
        for folder in DATASET_PATH.iterdir()
        if folder.is_dir() and len(list(folder.glob("*.jpg"))) >= MIN_IMAGES
    )
    class_names = [folder.name for folder in class_dirs]
    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    for class_dir in class_dirs:
        class_index = class_to_index[class_dir.name]
        for image_path in sorted(class_dir.glob("*.jpg"))[:MIN_IMAGES]:
            image_paths.append(str(image_path))
            labels.append(class_index)

    return image_paths, labels, class_names


def build_dataset(image_paths, labels, batch_size, preprocess_fn):
    """Cree le dataset TensorFlow pour la validation."""

    def parse_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        image = preprocess_fn(image)
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((
        tf.constant(image_paths, dtype=tf.string),
        tf.constant(labels, dtype=tf.int32),
    ))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def predict_model(model, dataset):
    """Retourne les vraies classes et les classes predites."""
    y_true = []
    y_pred = []

    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predictions.tolist())

    return np.array(y_true), np.array(y_pred)


def calculate_far_frr(conf_matrix, class_names):
    """
    Calcule FAR et FRR pour chaque classe selon une logique one-vs-all.

    FAR = FP / (FP + TN)
    FRR = FN / (FN + TP)
    TAR = TP / (TP + FN)
    """
    total = conf_matrix.sum()
    rows = []

    for idx, class_name in enumerate(class_names):
        tp = int(conf_matrix[idx, idx])
        fp = int(conf_matrix[:, idx].sum() - tp)
        fn = int(conf_matrix[idx, :].sum() - tp)
        tn = int(total - tp - fp - fn)

        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        tar = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        errors = fp + fn

        rows.append({
            "class_index": idx,
            "class_name": class_name.replace("_", " "),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "FAR": far,
            "FRR": frr,
            "TAR": tar,
            "precision": precision,
            "errors": errors,
            "support": int(conf_matrix[idx, :].sum()),
        })

    return pd.DataFrame(rows)


def select_top_classes(metrics_df, top_n):
    """Selectionne les classes les plus interessantes a afficher."""
    top = metrics_df.sort_values(
        by=["errors", "FRR", "FAR"],
        ascending=False,
    ).head(top_n)
    return top["class_index"].to_numpy()


def plot_confusion_matrix(conf_matrix, class_names, selected_classes, output_path):
    """Affiche une matrice de confusion lisible sur les classes selectionnees."""
    selected_matrix = conf_matrix[np.ix_(selected_classes, selected_classes)]
    selected_labels = [class_names[i].replace("_", " ") for i in selected_classes]

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        selected_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=selected_labels,
        yticklabels=selected_labels,
    )
    plt.title("Matrice de confusion")
    plt.xlabel("Classe predite")
    plt.ylabel("Classe reelle")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_correlation_matrix(metrics_df, output_path):
    """Affiche la matrice de correlation entre les metriques numeriques."""
    columns = ["TP", "FP", "FN", "TN", "FAR", "FRR", "TAR", "precision", "errors", "support"]
    correlation = metrics_df[columns].corr(method="pearson")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
    )
    plt.title("Matrice de correlation des metriques")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

    return correlation


def plot_far_frr(metrics_df, selected_classes, output_path):
    """Affiche FAR et FRR pour les classes selectionnees."""
    selected = metrics_df[metrics_df["class_index"].isin(selected_classes)].copy()
    selected = selected.sort_values("errors", ascending=False)

    x = np.arange(len(selected))
    width = 0.38

    plt.figure(figsize=(15, 7))
    plt.bar(x - width / 2, selected["FAR"] * 100, width, label="FAR (%)")
    plt.bar(x + width / 2, selected["FRR"] * 100, width, label="FRR (%)")
    plt.xticks(x, selected["class_name"], rotation=45, ha="right", fontsize=8)
    plt.ylabel("Taux (%)")
    plt.title("Comparaison FAR et FRR par classe")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def evaluate(model_name, batch_size, top_n):
    config = MODEL_CONFIG[model_name]
    model_path = config["model_path"]
    results_dir = config["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modele introuvable : {model_path}\n"
            f"Lance d'abord : python train_all.py --model {model_name}"
        )

    print(f"Chargement du modele : {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    print("Chargement du dataset...")
    image_paths, labels, class_names = collect_dataset()

    _, val_indices = train_test_split(
        range(len(image_paths)),
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    val_paths = [image_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    print(f"Images de validation : {len(val_paths)}")
    print(f"Nombre de classes    : {len(class_names)}")

    val_dataset = build_dataset(
        val_paths,
        val_labels,
        batch_size,
        config["preprocess"],
    )

    print("Prediction sur l'ensemble de validation...")
    y_true, y_pred = predict_model(model, val_dataset)

    print("Calcul de la matrice de confusion...")
    all_class_indices = np.arange(len(class_names))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=all_class_indices)

    print("Calcul FAR / FRR...")
    metrics_df = calculate_far_frr(conf_matrix, class_names)
    selected_classes = select_top_classes(metrics_df, top_n)

    confusion_path = results_dir / f"confusion_matrix_top{top_n}.png"
    correlation_path = results_dir / "correlation_matrix_metrics.png"
    far_frr_path = results_dir / f"far_frr_top{top_n}.png"
    csv_path = results_dir / "far_frr_by_class.csv"
    correlation_csv_path = results_dir / "correlation_matrix_metrics.csv"
    summary_path = results_dir / "evaluation_summary.json"

    print("Generation des graphiques...")
    plot_confusion_matrix(conf_matrix, class_names, selected_classes, confusion_path)
    correlation = plot_correlation_matrix(metrics_df, correlation_path)
    plot_far_frr(metrics_df, selected_classes, far_frr_path)

    metrics_df.to_csv(csv_path, index=False, encoding="utf-8")
    correlation.to_csv(correlation_csv_path, encoding="utf-8")

    accuracy = float(np.mean(y_true == y_pred))
    mean_far = float(metrics_df["FAR"].mean())
    mean_frr = float(metrics_df["FRR"].mean())
    corr_far_frr = float(metrics_df[["FAR", "FRR"]].corr().iloc[0, 1])

    summary = {
        "model": model_name,
        "num_classes": int(len(class_names)),
        "validation_images": int(len(y_true)),
        "accuracy": accuracy,
        "mean_FAR": mean_far,
        "mean_FRR": mean_frr,
        "correlation_FAR_FRR": corr_far_frr,
        "confusion_matrix_image": str(confusion_path),
        "correlation_matrix_image": str(correlation_path),
        "far_frr_image": str(far_frr_path),
        "far_frr_csv": str(csv_path),
        "correlation_csv": str(correlation_csv_path),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nEvaluation terminee")
    print(f"Accuracy             : {accuracy * 100:.2f}%")
    print(f"FAR moyen            : {mean_far * 100:.4f}%")
    print(f"FRR moyen            : {mean_frr * 100:.4f}%")
    print(f"Correlation FAR/FRR  : {corr_far_frr:.4f}")
    print(f"Matrice confusion    : {confusion_path}")
    print(f"Matrice correlation  : {correlation_path}")
    print(f"Graphique FAR/FRR    : {far_frr_path}")
    print(f"CSV FAR/FRR          : {csv_path}")
    print(f"Resume JSON          : {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Affiche matrice de confusion, matrice de correlation, FAR et FRR."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIG.keys()),
        default="efficientnet",
        help="Modele a evaluer.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Taille des batchs pendant la prediction.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Nombre de classes affichees dans les graphiques.",
    )
    args = parser.parse_args()
    evaluate(args.model, args.batch_size, args.top_n)


if __name__ == "__main__":
    main()
