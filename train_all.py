"""
Lanceur principal — entraine VGGFace, ArcFace et FaceNet sequentiellement.
Chaque modele est entraine separement et ses resultats sauvegardes dans results/<modele>/.

Usage:
    python train_all.py                  # entraine les 3 modeles
    python train_all.py --model vggface  # entraine un seul modele
    python train_all.py --model arcface
    python train_all.py --model facenet
"""
import argparse
import json
import time
import traceback
from pathlib import Path


def run_model(name: str) -> dict:
    if name == "vggface":
        from models.train_vggface import train
    elif name == "arcface":
        from models.train_arcface import train
    elif name == "facenet":
        from models.train_facenet import train
    elif name == "efficientnet":
        from models.train_efficientnet import train
    elif name == "resnet50v2":
        from models.train_resnet50v2 import train
    else:
        raise ValueError(f"Modele inconnu : {name}")
    return train()


def main():
    parser = argparse.ArgumentParser(description="Face Recognition Training Pipeline")
    parser.add_argument(
        "--model",
        choices=["vggface", "arcface", "facenet", "efficientnet", "resnet50v2", "all"],
        default="all",
        help="Modele a entrainer (defaut: all)",
    )
    args = parser.parse_args()

    models_to_run = (
        ["vggface", "arcface", "facenet", "efficientnet", "resnet50v2"]
        if args.model == "all" else [args.model]
    )

    print("=" * 60)
    print("  PIPELINE D'ENTRAINEMENT — RECONNAISSANCE FACIALE")
    print(f"  Modeles : {', '.join(m.upper() for m in models_to_run)}")
    print(f"  Dataset : LFW-Deepfunneled (>= 10 images/classe)")
    print("=" * 60)

    all_results = {}

    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"  MODELE : {model_name.upper()}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            res = run_model(model_name)
            elapsed = time.time() - t0
            res["training_time_sec"] = round(elapsed, 1)
            all_results[model_name] = res
            print(f"  >> {model_name.upper()} termine en {elapsed:.1f}s")
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  >> ERREUR sur {model_name.upper()} ({elapsed:.1f}s) : {exc}")
            traceback.print_exc()
            all_results[model_name] = {"error": str(exc)}

    # ── Sauvegarde du rapport global ──
    Path("results").mkdir(exist_ok=True)
    report_path = Path("results") / "all_results.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Affichage resume ──
    print("\n" + "=" * 60)
    print("  RESUME D'ENTRAINEMENT")
    print("=" * 60)
    for name, res in all_results.items():
        if "error" in res:
            print(f"  {name.upper():<10} ECHEC — {res['error']}")
        else:
            acc_key = "best_val_accuracy" if "best_val_accuracy" in res else "best_val_acc_nn"
            acc     = res.get(acc_key, res.get("val_accuracy", 0.0))
            nc      = res.get("num_classes", "?")
            ni      = res.get("total_images", "?")
            t       = res.get("training_time_sec", "?")
            print(
                f"  {name.upper():<10} Val Acc: {acc:.4f} | "
                f"Classes: {nc} | Images: {ni} | Temps: {t}s"
            )

    print(f"\n  Rapport complet -> {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
