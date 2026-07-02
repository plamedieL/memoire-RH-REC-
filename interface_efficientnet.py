"""
Systeme RH — Reconnaissance Faciale
Interface Gradio professionnelle : identification faciale + gestion arrivees/departs.
Backbone : EfficientNetB0

Lancement :
    python interface_efficientnet.py
"""
import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import gradio as gr
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

# ── Chemins ───────────────────────────────────────────────────────────────────
DATASET_PATH   = Path("data/lfw-deepfunneled/lfw-deepfunneled")
RESULTS_DIR    = Path("results/efficientnet")
MODEL_PATH     = RESULTS_DIR / "efficientnet_best.keras"
CLASS_JSON     = RESULTS_DIR / "class_names.json"
LOG_CSV        = Path("results/pointage_log_efficientnet.csv")
MIN_IMAGES     = 10
IMG_SIZE       = (224, 224)
TOP_K          = 5
CONF_THRESHOLD = 0.10

LOG_COLUMNS = ["Date", "Heure", "Employe", "Type", "Confiance (%)", "Statut"]

CSS = """
#header { text-align: center; background: linear-gradient(135deg,#1b5e20,#2e7d32);
          padding: 20px; border-radius: 12px; color: white; margin-bottom: 16px; }
#header h1 { color: white; margin: 0; font-size: 2rem; }
#header p  { color: #c8e6c9; margin: 4px 0 0; }
#clock     { text-align: center; font-size: 2rem; font-weight: bold;
             color: #1b5e20; letter-spacing: 4px; padding: 8px; }
#card      { border: 1px solid #e0e0e0; border-radius: 10px; padding: 16px;
             background: #fafafa; }
#result_name { font-size: 1.5rem; font-weight: bold; text-align: center;
               padding: 12px; border-radius: 8px; margin: 8px 0; }
#btn_arrive  { background: #2e7d32 !important; }
#btn_depart  { background: #c62828 !important; }
#stats       { display: flex; gap: 12px; justify-content: center; margin: 8px 0; }
.stat-card   { background: white; border-radius: 8px; padding: 12px 24px;
               text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,.1); flex: 1; }
.stat-num    { font-size: 2rem; font-weight: bold; }
.stat-lbl    { font-size: .85rem; color: #666; }
"""


# ── Modele ────────────────────────────────────────────────────────────────────

def load_class_names() -> list:
    if CLASS_JSON.exists():
        with open(CLASS_JSON) as f:
            return json.load(f)
    names = sorted(
        d.name for d in DATASET_PATH.iterdir()
        if d.is_dir() and len(list(d.glob("*.jpg"))) >= MIN_IMAGES
    )
    CLASS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_JSON, "w") as f:
        json.dump(names, f)
    return names


def load_model_and_classes():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modele introuvable : {MODEL_PATH} — "
            "lancez d'abord : python train_all.py --model efficientnet"
        )
    model = tf.keras.models.load_model(str(MODEL_PATH))
    return model, load_class_names()


print("Chargement du modele EfficientNetB0...")
try:
    MODEL, CLASS_NAMES = load_model_and_classes()
    MODEL_STATUS = f"Modele pret — {len(CLASS_NAMES)} employes"
    MODEL_OK = True
except FileNotFoundError as e:
    MODEL, CLASS_NAMES = None, []
    MODEL_STATUS = str(e)
    MODEL_OK = False
    print(f"[ERREUR] {e}")


# ── Preprocessing & Prediction ────────────────────────────────────────────────

def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict_face(image: Image.Image):
    if MODEL is None:
        return "Inconnu", 0.0, []
    x     = preprocess(image)
    probs = MODEL.predict(x, verbose=0)[0]
    top_k = np.argsort(probs)[::-1][:TOP_K]
    results = [(CLASS_NAMES[i].replace("_", " "), float(probs[i])) for i in top_k]
    name, conf = results[0]
    if conf < CONF_THRESHOLD:
        name = "Inconnu"
    return name, conf, results


# ── Dessin sur image ──────────────────────────────────────────────────────────

def draw_result(image: Image.Image, name: str, conf: float) -> Image.Image:
    img  = image.convert("RGBA").copy()
    w, h = img.size
    bh   = max(48, h // 7)

    band = Image.new("RGBA", (w, bh), (27, 94, 32, 210))
    img.paste(band, (0, 0), band)

    draw  = ImageDraw.Draw(img)
    color = (102, 255, 153) if name != "Inconnu" else (255, 100, 100)
    text  = f"  {name}   ({conf*100:.1f}%)"
    try:
        font = ImageFont.truetype("arial.ttf", size=max(18, bh // 2))
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, bh // 5), text, fill=color, font=font)
    return img.convert("RGB")


# ── Journal CSV ───────────────────────────────────────────────────────────────

def init_csv():
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_CSV.exists():
        with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LOG_COLUMNS)


def append_csv(row: dict):
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([row[c] for c in LOG_COLUMNS])


def load_log_df() -> pd.DataFrame:
    if LOG_CSV.exists():
        df = pd.read_csv(LOG_CSV, encoding="utf-8")
        if not df.empty:
            return df.iloc[::-1].reset_index(drop=True)
    return pd.DataFrame(columns=LOG_COLUMNS)


def stats_html(df: pd.DataFrame) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    today_df = df[df["Date"] == today] if not df.empty and "Date" in df.columns else pd.DataFrame()
    arrivees  = len(today_df[today_df["Type"] == "Arrivee"]) if not today_df.empty else 0
    departs   = len(today_df[today_df["Type"] == "Depart"])  if not today_df.empty else 0
    presents  = max(0, arrivees - departs)
    total     = len(today_df)
    return f"""
    <div id="stats">
      <div class="stat-card">
        <div class="stat-num" style="color:#2e7d32">{arrivees}</div>
        <div class="stat-lbl">Arrivees aujourd'hui</div>
      </div>
      <div class="stat-card">
        <div class="stat-num" style="color:#1565c0">{presents}</div>
        <div class="stat-lbl">Presents</div>
      </div>
      <div class="stat-card">
        <div class="stat-num" style="color:#c62828">{departs}</div>
        <div class="stat-lbl">Departs</div>
      </div>
      <div class="stat-card">
        <div class="stat-num" style="color:#6a1b9a">{total}</div>
        <div class="stat-lbl">Total pointages</div>
      </div>
    </div>
    """


init_csv()


# ── Fonctions Gradio ──────────────────────────────────────────────────────────

def identify(input_image):
    if input_image is None:
        return None, "", "Aucune image fournie.", gr.update(visible=False), gr.update(visible=False)

    pil_img = Image.fromarray(input_image)
    name, conf, top_results = predict_face(pil_img)
    annotated = draw_result(pil_img, name, conf)

    if name == "Inconnu":
        name_html = '<div id="result_name" style="background:#ffebee;color:#c62828">Inconnu — Acces refuse</div>'
        show_btns = False
    else:
        name_html = f'<div id="result_name" style="background:#e8f5e9;color:#2e7d32">{name}</div>'
        show_btns = True

    top_text = "\n".join(
        f"{'>>>' if i == 0 else '   '} {r[0]:<30} {r[1]*100:5.2f}%"
        for i, r in enumerate(top_results)
    )

    return (
        np.array(annotated),
        name_html,
        top_text,
        gr.update(visible=show_btns, value=f"Enregistrer Arrivee — {name}"),
        gr.update(visible=show_btns, value=f"Enregistrer Depart  — {name}"),
    )


def pointer(input_image, type_pointage: str):
    if input_image is None:
        return load_log_df(), stats_html(load_log_df()), "Aucune image pour le pointage."

    pil_img  = Image.fromarray(input_image)
    name, conf, _ = predict_face(pil_img)

    if name == "Inconnu":
        df = load_log_df()
        return df, stats_html(df), "Employe non identifie — pointage refuse."

    now  = datetime.now()
    row  = {
        "Date":          now.strftime("%Y-%m-%d"),
        "Heure":         now.strftime("%H:%M:%S"),
        "Employe":       name,
        "Type":          type_pointage,
        "Confiance (%)": f"{conf*100:.1f}",
        "Statut":        "OK",
    }
    append_csv(row)

    df    = load_log_df()
    emoji = "Arrivee" if type_pointage == "Arrivee" else "Depart"
    msg   = f"{emoji} enregistre pour {name} a {row['Heure']}"
    return df, stats_html(df), msg


def pointer_arrivee(img):
    return pointer(img, "Arrivee")


def pointer_depart(img):
    return pointer(img, "Depart")


def refresh_log():
    df = load_log_df()
    return df, stats_html(df)


def get_clock():
    return datetime.now().strftime("%H:%M:%S")


def export_csv():
    return str(LOG_CSV) if LOG_CSV.exists() else None


# ── Interface Gradio ──────────────────────────────────────────────────────────

with gr.Blocks(css=CSS, title="Systeme RH — Reconnaissance Faciale") as demo:

    # ── En-tete ──
    gr.HTML(f"""
    <div id="header">
      <h1>Systeme RH — Reconnaissance Faciale</h1>
      <p>Reconnaissance faciale par EfficientNetB0 — {MODEL_STATUS}</p>
    </div>
    """)

    # ── Horloge ──
    with gr.Row():
        clock_display = gr.Textbox(
            value=get_clock(),
            label="Heure actuelle",
            interactive=False,
            elem_id="clock",
            scale=1,
        )
        date_display = gr.Textbox(
            value=datetime.now().strftime("%A %d %B %Y"),
            label="Date",
            interactive=False,
            scale=2,
        )

    gr.HTML("<hr/>")

    # ── Zone principale : identification ──
    with gr.Row():

        # Colonne gauche : camera
        with gr.Column(scale=1, elem_id="card"):
            gr.Markdown("### Capture")
            input_img = gr.Image(
                label="Photo / Webcam",
                sources=["upload", "webcam"],
                type="numpy",
                height=300,
            )
            btn_identify = gr.Button("Identifier", variant="primary", size="lg")

        # Colonne droite : resultat
        with gr.Column(scale=1, elem_id="card"):
            gr.Markdown("### Identification")
            output_img = gr.Image(label="Resultat", type="numpy", height=220)
            name_html  = gr.HTML('<div id="result_name" style="background:#f5f5f5;color:#555">En attente...</div>')
            top_text   = gr.Textbox(label="Top predictions", lines=5, interactive=False)

            with gr.Row():
                btn_arrive = gr.Button(
                    "Enregistrer Arrivee",
                    variant="primary",
                    visible=False,
                    elem_id="btn_arrive",
                )
                btn_depart = gr.Button(
                    "Enregistrer Depart",
                    variant="stop",
                    visible=False,
                    elem_id="btn_depart",
                )
            msg_box = gr.Textbox(label="Message", interactive=False, lines=1)

    gr.HTML("<hr/>")

    # ── Journal de pointage ──
    gr.Markdown("### Journal de pointage")

    stats_box = gr.HTML(stats_html(load_log_df()))

    log_table = gr.Dataframe(
        value=load_log_df(),
        headers=LOG_COLUMNS,
        interactive=False,
        wrap=True,
        row_count=10,
    )

    with gr.Row():
        btn_refresh = gr.Button("Actualiser le journal", size="sm")
        btn_export  = gr.DownloadButton(
            "Exporter CSV",
            value=export_csv,
            size="sm",
        )

    # ── Evenements ──

    btn_identify.click(
        fn=identify,
        inputs=input_img,
        outputs=[output_img, name_html, top_text, btn_arrive, btn_depart],
    )

    btn_arrive.click(
        fn=pointer_arrivee,
        inputs=input_img,
        outputs=[log_table, stats_box, msg_box],
    )

    btn_depart.click(
        fn=pointer_depart,
        inputs=input_img,
        outputs=[log_table, stats_box, msg_box],
    )

    btn_refresh.click(
        fn=refresh_log,
        outputs=[log_table, stats_box],
    )

    try:
        timer = gr.Timer(value=1)
        timer.tick(fn=get_clock, outputs=clock_display)
    except Exception:
        pass


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        inbrowser=True,
    )
