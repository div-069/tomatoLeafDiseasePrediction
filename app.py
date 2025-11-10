from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
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

# -----------------------------
# 📦 Load Trained Model
# -----------------------------
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

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        img_path=img_path
    )

# -----------------------------
# 📊 Model Evaluation Section
# -----------------------------
def evaluate_model():
    print("\n🔎 Evaluating model performance...")

    # Detect validation or test folder
    if os.path.exists(os.path.join(DATASET_PATH, "test")):
        eval_dir = os.path.join(DATASET_PATH, "test")
    elif os.path.exists(os.path.join(DATASET_PATH, "val")):
        eval_dir = os.path.join(DATASET_PATH, "val")
    else:
        print("⚠️ No 'test' or 'val' directory found. Skipping evaluation.")
        return

    img_size = (224, 224)
    batch_size = 32

    # ✅ One-hot encoded labels to match model output
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        eval_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        label_mode="categorical"
    )

    class_names_eval = test_ds.class_names

    # Normalize dataset
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.AUTOTUNE
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    # Evaluate model
    loss, accuracy = model.evaluate(test_ds)
    print(f"\n✅ Model Accuracy on Validation Set: {accuracy * 100:.2f}%")

    # Get predictions
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names_eval, digits=4)
    print("\n🔍 Detailed Classification Report:")
    print(report)

    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0)

    print("\n📊 Per-Class Metrics:")
    for c, p, r, f in zip(class_names_eval, precision, recall, f1):
        print(f"{c:40s} | Precision: {p:.3f} | Recall: {r:.3f} | F1: {f:.3f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_eval,
                yticklabels=class_names_eval)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('🧩 Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # Plot Precision, Recall, F1
    metrics = ['Precision', 'Recall', 'F1-score']
    values = [precision, recall, f1]
    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        plt.bar(np.arange(len(class_names_eval)) + i*0.25, values[i], width=0.25, label=metric)

    plt.xticks(np.arange(len(class_names_eval)) + 0.25, class_names_eval, rotation=45, ha='right')
    plt.ylabel("Score")
    plt.title("📈 Per-Class Precision, Recall & F1-score")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# 🚀 Run Flask App
# -----------------------------
if __name__ == "__main__":
    # Run evaluation before starting Flask app
    evaluate_model()

    # Start Flask
    app.run(debug=True)
