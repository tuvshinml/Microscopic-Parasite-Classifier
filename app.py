import os, logging
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.environ.get("MODEL_PATH",    os.path.join(BASE_DIR, "parasitenet_final.keras"))
MODEL_PATH_H5 = os.environ.get("MODEL_PATH_H5", os.path.join(BASE_DIR, "parasitenet.h5"))
IMG_SIZE      = int(os.environ.get("IMG_SIZE", 224))
PORT          = int(os.environ.get("PORT", 10000))

CLASS_NAMES = [
    "Ascaris lumbricoides","Capillaria philippinensis","Echinococcus granulosus",
    "Enterobius vermicularis","Fasciolopsis buski","Hookworm egg",
    "Hymenolepis diminuta","Hymenolepis nana","Opisthorchis viverrine",
    "Paragonimus spp","Taenia spp. egg","Trichuris trichiura",
]

model = None

def load_model():
    global model
    import tensorflow as tf
    path = MODEL_PATH if os.path.isfile(MODEL_PATH) else MODEL_PATH_H5
    if not os.path.isfile(path):
        log.warning("Model файл олдсонгүй: %s / %s", MODEL_PATH, MODEL_PATH_H5)
        return
    log.info("Model ачааллаж байна: %s", path)
    model = tf.keras.models.load_model(path)
    log.info("Model амжилттай ачаалагдлаа.")

def preprocess(image_bytes):
    import cv2
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Зургийг уншиж чадсангүй.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return np.expand_dims(img.astype(np.float32), axis=0)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"]  = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

load_model()

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status":"ok","model_loaded":model is not None,
                    "img_size":IMG_SIZE,"classes":len(CLASS_NAMES),"class_names":CLASS_NAMES})

@app.route("/predict", methods=["POST","OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204
    file = request.files.get("file") or request.files.get("image")
    if file is None:
        return jsonify({"error":"'file' эсвэл 'image' field шаардлагатай."}), 400
    try:
        x = preprocess(file.read())
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    if model is None:
        return jsonify({"error":"Загвар ачаалагдаагүй."}), 503
    preds = model.predict(x, verbose=0)[0].astype(float)
    if preds.min() < 0 or preds.max() > 1 or abs(preds.sum()-1.0) > 0.05:
        e = np.exp(preds - preds.max())
        preds = e / e.sum()
    top_idx  = int(np.argmax(preds))
    top3_idx = np.argsort(preds)[::-1][:3].tolist()
    top3 = [{"rank":i+1,"class_index":idx,"class_name":CLASS_NAMES[idx],
              "confidence":round(float(preds[idx]),6)} for i,idx in enumerate(top3_idx)]
    return jsonify({"class_index":top_idx,"class_name":CLASS_NAMES[top_idx],
                    "confidence":round(float(preds[top_idx]),6),
                    "probabilities":[round(float(p),6) for p in preds],"top3":top3})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
