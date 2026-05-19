import os
import logging
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_PATH = r"C:/Users/Dell/Documents/P2_SpringerNature_Manuscript/parasitenet_final.keras"
MODEL_PATH_H5 = r"C:/Users/Dell/Documents/P2_SpringerNature_Manuscript/parasitenet_final.h5"

IMG_SIZE = 224
PORT = 5000

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

model = None

def load_model():
    global model
    import tensorflow as tf

    path = MODEL_PATH if os.path.isfile(MODEL_PATH) else MODEL_PATH_H5
    model = tf.keras.models.load_model(path)

def preprocess(image_bytes):
    import cv2

    nparr = np.frombuffer(image_bytes, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    arr = img.astype(np.float32)

    return np.expand_dims(arr, axis=0)

app = Flask(__name__)

CORS(app)

@app.route("/", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "img_size": IMG_SIZE,
        "classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
    })

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files.get("file") or request.files.get("image")

    x = preprocess(file.read())

    preds = model.predict(x, verbose=0)[0].astype(float)

    if preds.min() < 0 or preds.max() > 1 or abs(preds.sum() - 1.0) > 0.05:
        e = np.exp(preds - preds.max())
        preds = e / e.sum()

    class_index = int(np.argmax(preds))

    top3_idx = np.argsort(preds)[::-1][:3].tolist()

    top3 = [
        {
            "rank": i + 1,
            "class_index": idx,
            "class_name": CLASS_NAMES[idx],
            "confidence": round(float(preds[idx]), 6),
        }
        for i, idx in enumerate(top3_idx)
    ]

    return jsonify({
        "class_index": class_index,
        "class_name": CLASS_NAMES[class_index],
        "confidence": round(float(preds[class_index]), 6),
        "probabilities": [round(float(p), 6) for p in preds],
        "top3": top3,
    })

if __name__ == "__main__":
    load_model()
    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False
    )