import os
# Reduce TensorFlow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pickle
import cv2
import io
from rembg import remove
from flask_cors import CORS

# --- Flask App ---
app = Flask(__name__)
CORS(app)

# --- Lazy-loading placeholders ---
model = None
lb = None

def load_model_once():
    """Load model and label binarizer only on first request."""
    global model, lb
    if model is None or lb is None:
        print("--- Loading model and label binarizer ---")
        model = load_model("pokedex.keras")
        with open("lb.pickle", "rb") as f:
            lb = pickle.load(f)
        print("--- Model loaded successfully ---")

# --- Image Preprocessing ---
def preprocess_image(pil_image):
    image = pil_image.convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (96, 96))
    image = image.astype("float32") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(pil_image):
    load_model_once()
    processed = preprocess_image(pil_image)
    probs = model.predict(processed, verbose=0)[0]
    idx = np.argmax(probs)
    label = lb.classes_[idx]
    confidence = float(probs[idx]) * 100
    return label, confidence

def remove_background_and_crop(pil_image):
    """Remove background and crop the result."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    input_data = buf.getvalue()
    output_data = remove(input_data)
    bg_removed = Image.open(io.BytesIO(output_data))
    bbox = bg_removed.getbbox()
    return bg_removed.crop(bbox) if bbox else bg_removed

# --- API Endpoint ---
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        original = Image.open(file.stream)

        # 1. Predict original image
        label1, confidence1 = predict_image(original)
        if confidence1 >= 90:
            return jsonify({"label": label1, "confidence": round(confidence1, 2), "used": "original"})

        # 2. Predict background removed image
        bg_removed_image = remove_background_and_crop(original)
        label2, confidence2 = predict_image(bg_removed_image)

        if confidence2 > confidence1:
            return jsonify({"label": label2, "confidence": round(confidence2, 2), "used": "background_removed"})
        else:
            return jsonify({"label": label1, "confidence": round(confidence1, 2), "used": "original"})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "An internal error occurred during processing."}), 500

# --- Run App ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
