from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import io
from PIL import Image

# ---------------- Setup ----------------
app = Flask(__name__)

# Load your trained model
MODEL_PATH = "trained_model_DNN1.h5"
model = load_model(MODEL_PATH)


CLASS_NAMES = ["Benign", "Malignant"]  
INPUT_SIZE = (224, 224)

# ---------------- Utils ----------------
def preprocess_image(image_bytes):
    """Convert uploaded image into tensor suitable for model"""
    img = Image.open(io.BytesIO(image_bytes)).resize(INPUT_SIZE)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)  # batch dimension
    x = x.astype(np.float32) / 255.0
    return x

# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Breast Cancer Detection Flask API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_bytes = file.read()
        tensor = preprocess_image(image_bytes)
        preds = model.predict(tensor)

        if preds.ndim == 2 and preds.shape[1] > 1:
            # multi-class
            idx = int(np.argmax(preds, axis=1)[0])
            confidence = float(np.max(preds, axis=1)[0])
        else:
            # binary
            p1 = float(preds[0][0]) if preds.ndim == 2 else float(preds[0])
            idx = int(p1 >= 0.5)
            confidence = p1 if idx == 1 else (1.0 - p1)

        label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class_{idx}"

        return jsonify({
            "predicted_class": label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
