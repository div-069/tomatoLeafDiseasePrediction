# 🍅 Tomato Leaf Disease Prediction (Flask + TensorFlow)
A Flask web application that detects tomato leaf diseases using a pre-trained deep learning (CNN) model. It also includes a model evaluation pipeline with detailed performance metrics and visualizations.

## 📸 Features
- Upload tomato leaf images and get real-time disease predictions  
- Displays predicted class and confidence score  
- Evaluates model performance on test/validation datasets  
- Automatically generates: classification report, confusion matrix, per-class metrics  
- Built with TensorFlow / Keras, Flask, Matplotlib, Seaborn, and Scikit-learn

## 🧱 Project Structure
tomatoLeafDiseasePrediction/  
│  
├── app.py — Flask backend  
├── tomato_model.h5 — Trained CNN model  
├── tomato_dataset/ — Dataset (train/test/val)  
│   ├── train/  
│   ├── test/ or val/  
│   └── ...  
├── static/uploads — Uploaded images  
└── templates/index.html — Web interface  

## ⚙️ Installation & Setup
1️⃣ Clone the repository  
git clone https://github.com/div-069/tomatoLeafDiseasePrediction.git  
cd tomatoLeafDiseasePrediction  

2️⃣ Create & activate a virtual environment  
python -m venv venv  
venv\Scripts\activate  (Windows)  
source venv/bin/activate  (Mac/Linux)  

3️⃣ Install dependencies  
pip install tensorflow flask numpy scikit-learn matplotlib seaborn  
(Optional) For GPU acceleration: pip install tensorflow-gpu  

## 🧠 Model File
Place your trained model file `tomato_model.h5` in the project root directory. The model should output probabilities for each tomato disease class.  
Dataset structure example:  
train → Bacterial_spot, Early_blight, Healthy, etc.  
test/val → Bacterial_spot, Early_blight, Healthy, etc.  

## 🚀 Run the Application
1️⃣ Start the Flask server  
python app.py  
2️⃣ Open your browser and go to http://127.0.0.1:5000/  
3️⃣ Upload a tomato leaf image — the app predicts the disease class and confidence score.

## 🔍 Model Evaluation
Before launching the app, `app.py` automatically runs `evaluate_model()` which:  
- Loads the validation/test dataset  
- Evaluates model accuracy  
- Prints classification metrics  
- Displays confusion matrix and performance graphs  

Example output:  
Model Accuracy: 96.45%  
Bacterial_spot — Precision 0.97, Recall 0.95, F1 0.96  
Early_blight — Precision 0.95, Recall 0.97, F1 0.96  
Healthy — Precision 0.99, Recall 0.98, F1 0.98  

## 📊 Visualization Outputs
1️⃣ Confusion Matrix — Heatmap of predicted vs actual classes  
2️⃣ Per-Class Metrics — Precision, Recall, F1-score bar plots  
3️⃣ Example Prediction — Prediction: Early Blight | Confidence: 98.7%  

## 🧮 Technologies Used
Backend: Flask  
Deep Learning: TensorFlow / Keras  
Evaluation: Scikit-learn  
Visualization: Matplotlib, Seaborn  
Frontend: HTML + Jinja2  

## 🧑‍💻 Author
**Divyanshu Chaudhary**  
GitHub: [div-069](https://github.com/div-069)

## 🧾 License
This project is licensed under the MIT License. Feel free to use, modify, and distribute it.
