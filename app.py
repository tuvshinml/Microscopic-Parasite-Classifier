import os
import logging
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
# Замыг env variable-аас уншина; байхгүй бол ажлын хавтасны файлыг хайна
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "parasitenet_final.keras"),
)
MODEL_PATH_H5 = os.environ.get(
    "MODEL_PATH_H5",
    os.path.join(os.path.dirname(__file__), "parasitenet.h5"),
)
IMG_SIZE = int(os.environ.get("IMG_SIZE", 224))
PORT     = int(os.environ.get("PORT", 5000))

CLASS_NAMES = [
    "Ascaris lumbricoides",
    "Capillaria philippinensis",
    "Echinococcus granulosus",
    "Enterobius vermicularis",
    "Fasciolopsis buski",
    "Hookworm egg",
    "Hymenolepis diminuta",
    "Hymenolepis nana",
    "Opisthorchis viverrine",
    "Paragonimus spp",
    "Taenia spp. egg",
    "Trichuris trichiura",
]

# ── Model ──────────────────────────────────────────────────────────────────────
model = None

def load_model():
    """Model-ийг disk-ээс ачаална (.keras эсвэл .h5)."""
    global model
    import tensorflow as tf

    path = MODEL_PATH if os.path.isfile(MODEL_PATH) else MODEL_PATH_H5
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model файл олдсонгүй.\n"
            f"  Хайсан замууд:\n  • {MODEL_PATH}\n  • {MODEL_PATH_H5}\n"
            "MODEL_PATH env variable тохируулна уу."
        )

    log.info("Model ачааллаж байна: %s", path)
    model = tf.keras.models.load_model(path)
    log.info("Model амжилттай ачаалагдлаа.")

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    """
    Зургийг загварт оруулах хэлбэрт оруулна.
    Буцаах утга: (1, IMG_SIZE, IMG_SIZE, 3) хэлбэртэй float32 array.
    """
    import cv2

    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Зургийг уншиж чадсангүй. Зөвхөн JPEG / PNG дэмжинэ.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    arr = img.astype(np.float32)          # загвар дотроо normalize хийдэг тул
    return np.expand_dims(arr, axis=0)   # /255 хийхгүй

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": model is not None,
        "img_size":     IMG_SIZE,
        "classes":      len(CLASS_NAMES),
        "class_names":  CLASS_NAMES,
    })

@app.route("/predict", methods=["POST"])
def predict():
    # ── 1. Файл хүлээн авах ──
    file = request.files.get("file") or request.files.get("image")
    if file is None:
        return jsonify({"error": "'file' эсвэл 'image' field шаардлагатай."}), 400

    # ── 2. Preprocessing ──
    try:
        x = preprocess(file.read())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422

    # ── 3. Загвар ачаалагдсан эсэх ──
    if model is None:
        return jsonify({"error": "Загвар ачаалагдаагүй байна."}), 503

    # ── 4. Таамаглал ──
    preds = model.predict(x, verbose=0)[0].astype(float)

    # Гаралт нь log-probability байвал softmax хэрэглэнэ
    if preds.min() < 0 or preds.max() > 1 or abs(preds.sum() - 1.0) > 0.05:
        e     = np.exp(preds - preds.max())
        preds = e / e.sum()

    class_index = int(np.argmax(preds))
    top3_idx    = np.argsort(preds)[::-1][:3].tolist()
    top3        = [
        {
            "rank":         i + 1,
            "class_index":  idx,
            "class_name":   CLASS_NAMES[idx],
            "confidence":   round(float(preds[idx]), 6),
        }
        for i, idx in enumerate(top3_idx)
    ]

    return jsonify({
        "class_index":  class_index,
        "class_name":   CLASS_NAMES[class_index],
        "confidence":   round(float(preds[class_index]), 6),
        "probabilities": [round(float(p), 6) for p in preds],
        "top3":         top3,
    })

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=PORT, debug=False)
