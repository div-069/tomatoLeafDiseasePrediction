# 🍅 Tomato Leaf Disease Prediction (Flask + TensorFlow)

A **Flask web application** that detects **tomato leaf diseases** using a pre-trained **deep learning (CNN)** model.  
It also includes a **model evaluation pipeline** with detailed performance metrics and visualizations.

---

## 📸 Features

✅ Upload tomato leaf images and get real-time disease predictions  
✅ Displays **predicted class** and **confidence score**  
✅ Evaluates model performance on test/validation datasets  
✅ Automatically generates:
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix (heatmap)
- Per-class metrics visualization  
✅ Built with **TensorFlow / Keras**, **Flask**, **Matplotlib**, **Seaborn**, and **Scikit-learn**

---

## 🧱 Project Structure

tomatoLeafDiseasePrediction/
│
├── app.py # Flask backend
├── tomato_model.h5 # Trained CNN model
├── tomato_dataset/ # Dataset (train/test/val)
│ ├── train/
│ ├── test/ or val/
│ └── ...
├── static/
│ └── uploads/ # Uploaded images
└── templates/
└── index.html # Web interface


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/div-069/tomatoLeafDiseasePrediction.git
cd tomatoLeafDiseasePrediction

2️⃣ Create & activate a virtual environment
bash
Copy code
python -m venv venv
venv\Scripts\activate      # For Windows
# source venv/bin/activate # For Mac/Linux

3️⃣ Install dependencies
bash
Copy code
pip install tensorflow flask numpy scikit-learn matplotlib seaborn

(Optional:) For GPU acceleration

pip install tensorflow-gpu

🧠 Model File
Place your trained model file tomato_model.h5 in the project root directory.

The model should output probabilities for each tomato disease class.

Expected dataset structure:

bash
Copy code
tomato_dataset/
├── train/
│   ├── Bacterial_spot/
│   ├── Early_blight/
│   ├── Healthy/
│   └── ...
├── test/  (or val/)
│   ├── Bacterial_spot/
│   ├── Early_blight/
│   ├── Healthy/
│   └── ...
🚀 Run the Application
1️⃣ Start the Flask server
bash
Copy code
python app.py
2️⃣ Open in browser
cpp
Copy code
http://127.0.0.1:5000/
3️⃣ Upload an image
Upload a tomato leaf image — the app predicts the disease class and confidence score.

🔍 Model Evaluation
Before launching the app, app.py automatically runs:

python
Copy code
evaluate_model()
This function:

Loads the validation/test dataset

Evaluates model accuracy

Prints classification metrics

Displays confusion matrix and performance graphs

Example output:

python-repl
Copy code
✅ Model Accuracy on Validation Set: 96.45%

🔍 Classification Report:
                 precision    recall  f1-score   support
Bacterial_spot       0.97      0.95      0.96       100
Early_blight         0.95      0.97      0.96       100
Healthy              0.99      0.98      0.98       100
...
📊 Visualization Outputs
1️⃣ Confusion Matrix

Heatmap of predicted vs actual disease classes

2️⃣ Per-Class Metrics

Bar plots showing Precision, Recall, and F1-score for each class

3️⃣ Example Prediction

makefile
Copy code
Prediction: Early Blight  
Confidence: 98.7%
🧮 Technologies Used
Component	Library
Backend	Flask
Deep Learning	TensorFlow / Keras
Evaluation	Scikit-learn
Visualization	Matplotlib, Seaborn
Frontend	HTML + Jinja2 (Flask templates)

🧑‍💻 Author
Divyanshu Chaudhary
🔗 GitHub: div-069

---

✅ **Done:**  