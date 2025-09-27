from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("tomato_model.h5")

# Load class names from dataset
class_names = sorted(os.listdir("tomato_dataset/tomato/train"))

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    img_path = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)

            # Preprocess image
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Prediction
            pred = model.predict(img_array)[0]
            pred_idx = np.argmax(pred)
            prediction = class_names[pred_idx]
            confidence = round(pred[pred_idx] * 100, 2)

    return render_template("index.html", prediction=prediction, confidence=confidence, img_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)
