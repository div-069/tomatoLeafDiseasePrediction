# -*- coding: utf-8 -*-
"""Tomato Leaf Disease Predition using CNN.ipynb

The original file is located somewhere in the world.

"""

from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets list -s tomato

!kaggle datasets download -d kaustubhb999/tomatoleaf

import zipfile

with zipfile.ZipFile("tomatoleaf.zip", 'r') as zip_ref:
    zip_ref.extractall("/content/tomato_dataset")

import os
print(os.listdir("/content/tomato_dataset"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

def count_images(directory):
  categories_count ={}
  categories = os.listdir(directory)
  for category in categories:
    category_dir = os.path.join(directory, category)
    image_count = len(os.listdir(category_dir))
    categories_count[category] = image_count
  return categories_count

train_dir = '/content/tomato_dataset/tomato/train'
validation_dir = '/content/tomato_dataset/tomato/val'

train_count = count_images(train_dir)
validation_count = count_images(validation_dir)

print("The train sample count: ", train_count)
print("The validation sample count: ", validation_count)

train_df = pd.DataFrame(list(train_count.items()), columns = ["Diseases", "Train Count"])
val_df = pd.DataFrame(list(validation_count.items()), columns = ["Diseases", "Validation Count"])

df = pd.merge(train_df, val_df, on="Diseases")
df

df_melted = df.melt(
    id_vars='Diseases',
    value_vars=['Train Count','Validation Count'],
    var_name='Dataset Split',
    value_name='Image Count'
)
plt.figure(figsize=(12,6))
sns.barplot(data = df_melted, x = "Diseases", y = "Image Count", hue = "Dataset Split")
plt.xticks(rotation = 45, ha="right")
plt.xlabel("Diseases")
plt.ylabel("Image Count")
plt.title("Image per Disease")
plt.show()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True,
    validation_split = 0.2
)

validation_datagen = ImageDataGenerator(
    rescale = 1./255,
    validation_split = 0.2
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'categorical',
    subset = 'training',
    shuffle = True,
    seed = 42
)

validation_data = validation_datagen.flow_from_directory(
    train_dir,
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'categorical',
    subset = 'validation',
    shuffle = True,
    seed = 42
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation = 'relu', input_shape = (224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation = 'relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation = 'relu'),
    MaxPooling2D(2,2),
    Conv2D(256, (3,3), activation = 'relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation = 'relu'),
    Dropout(0.5),
    Dense(10, activation = 'softmax')
    ])

model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)

model.summary()

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor = 'val_loss',
    patience = 5,
    restore_best_weights = True
)

history = model.fit(
    train_data,
    epochs = 8,
    validation_data = validation_data,
    callbacks = [early_stopping]
)

train_loss, train_acc = model.evaluate(train_data)
print("The train accuracy: ", train_acc*100)

val_loss, val_acc = model.evaluate(validation_data)
print("The validation accuracy: ", val_acc*100)

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Train Accuracy")
plt.plot(epochs_range, val_acc, label="Val Accuracy")
plt.legend()
plt.title("Accuracy Over Epochs")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label="Val Loss")
plt.legend()
plt.title("Loss Over Epochs")

plt.show()

# ✅ Save the trained model
model.save("tomato_model.h5")
print("✅ Model saved as tomato_model.h5")

# # ✅ Load the model back when needed
# from tensorflow.keras.models import load_model
# loaded_model = load_model("tomato_model.h5")
# print("🔁 Model loaded successfully!")

## zipping it cause wanted it locally
# !zip -r project_files.zip cnn_train.py tomato_dataset tomato_model.h5
# from google.colab import files
# files.download("project_files.zip")

