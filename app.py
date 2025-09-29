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

# Initialize Flask
app = Flask(__name__)
CORS(app)  # Optional, for React Native/Expo

# Load model and label binarizer once at startup
model = load_model("pokedex.keras")
with open("lb.pickle", "rb") as f:
    lb = pickle.load(f)

def preprocess_image(pil_image):
    image = pil_image.convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 🔥 Match original behavior
    image = cv2.resize(image, (96, 96))
    image = image.astype("float32") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


def predict_image(pil_image):
    processed = preprocess_image(pil_image)
    probs = model.predict(processed, verbose=0)[0]
    idx = np.argmax(probs)
    label = lb.classes_[idx]
    confidence = float(probs[idx]) * 100
    return label, confidence

def remove_background_and_crop(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    input_data = buf.getvalue()
    output_data = remove(input_data)
    bg_removed = Image.open(io.BytesIO(output_data))

    # Crop to non-transparent area
    bbox = bg_removed.getbbox()
    if bbox:
        cropped = bg_removed.crop(bbox)
    else:
        cropped = bg_removed
    return cropped

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Load original image
        original = Image.open(file.stream)

        # First prediction
        label1, confidence1 = predict_image(original)

        # If confident enough, return it
        if confidence1 >= 90:
            return jsonify({
                "label": label1,
                "confidence": confidence1,
                "used": "original"
            })

        # Try again with background removed
        bg_removed_image = remove_background_and_crop(original)
        label2, confidence2 = predict_image(bg_removed_image)

        print(label1, label2)
        print(confidence1, confidence2)

        # Return best prediction
        if confidence2 > confidence1:
            return jsonify({
                "label": label2,
                "confidence": confidence2,
                "used": "background_removed"
            })
        else:
            return jsonify({
                "label": label1,
                "confidence": confidence1,
                "used": "original"
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
