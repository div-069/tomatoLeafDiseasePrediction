import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Model loading
model = tf.keras.models.load_model("tomato_model.h5")
print("✅ Model loaded successfully!")

# Test image 
img_path = "testleaf.JPG"   

# Image preprocessing 
img = image.load_img(img_path, target_size=(224, 224))  
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Prediction
pred = model.predict(img_array)
predicted_class = np.argmax(pred, axis=1)

print("🔮 Prediction:", predicted_class)