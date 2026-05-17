# Install libraries
!pip install tensorflow deepface numpy opencv-python

# Import libraries
import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model

# Load and Display Sample Image
#insert your image path which will be downloaded from the dataset folder
img_path = "/content/database/close_eye/s0001_00001_0_0_0_0_0_01.png"
img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img_array is None:
    print("Error: Could not load image. Please check the file path.")
else:
    plt.imshow(img_array, cmap='gray')
    plt.show()

# Data Preparation
#insert the path of the dataset you downloaded
datadirectory = "/content/dataset"
classes = ['close_eye', 'open_eye']
img_size = 224
training_data = []

def create_training_data():
    for category in classes:
        path = os.path.join(datadirectory, category)
        class_num = classes.index(category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)  # Use RGB
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                print(f"Error loading image {img}: {e}")

create_training_data()
print("Total images loaded:", len(training_data))

# Shuffle dataset
import random
random.shuffle(training_data)

# Separate features and labels
X, y = [], []
for features, label in training_data:
    X.append(features)
    y.append(label)

# Convert to NumPy arrays
X = np.array(X).reshape(-1, img_size, img_size, 3)  # Ensure RGB channels
X = X / 255.0  # Normalize pixel values
y = np.array(y)

# Save dataset using pickle
with open("X.pickle", "wb") as pickle_out:
    pickle.dump(X, pickle_out)

with open("y.pickle", "wb") as pickle_out:
    pickle.dump(y, pickle_out)

# Train-test split (before training)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)

# Build Model
base_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
flat_layer = layers.Flatten()(base_model.output)
final_output = layers.Dense(1, activation='sigmoid')(flat_layer)

model = Model(inputs=base_model.input, outputs=final_output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train Model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)
