from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# ⚙️ Flask App Configuration
# -----------------------------
app = Flask(__name__)

MODEL_PATH = "tomato_model.h5"
DATASET_PATH = "tomato_dataset/tomato"
UPLOAD_FOLDER = "static/uploads"

# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model
print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Load class names (from train directory)
class_names = sorted(os.listdir(os.path.join(DATASET_PATH, "train")))
print("📂 Classes:", class_names)


# -----------------------------
# 🌿 Flask Route for Prediction
# -----------------------------
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

            # Make prediction
            pred = model.predict(img_array)[0]
            pred_idx = np.argmax(pred)
            prediction = class_names[pred_idx]
            confidence = round(pred[pred_idx] * 100, 2)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           img_path=img_path)


# -----------------------------
# 📊 Model Evaluation Section
# -----------------------------
def evaluate_model():
    print("\n🔎 Evaluating model performance...")

    # Detect whether "test" or "val" folder exists
    if os.path.exists(os.path.join(DATASET_PATH, "test")):
        eval_dir = os.path.join(DATASET_PATH, "test")
    elif os.path.exists(os.path.join(DATASET_PATH, "val")):
        eval_dir = os.path.join(DATASET_PATH, "val")
    else:
        print("⚠️ No 'test' or 'val' directory found. Skipping evaluation.")
        return

    img_size = (224, 224)
    batch_size = 32

    # Load evaluation dataset
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        eval_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        label_mode="categorical"   # ✅ Ensures same shape as model output
    )

    # Evaluate model
    loss, accuracy = model.evaluate(test_ds)
    print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")

    # Get true & predicted labels
    y_true = np.concatenate([np.argmax(y, axis=1) for x, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=test_ds.class_names)
    print("\n🔍 Classification Report:")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_ds.class_names,
                yticklabels=test_ds.class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('🧩 Confusion Matrix')
    plt.show()


# -----------------------------
# 🚀 Run Flask App
# -----------------------------
if __name__ == "__main__":
    # Run evaluation once before starting server
    evaluate_model()

    # Then start Flask app
    app.run(debug=True)
