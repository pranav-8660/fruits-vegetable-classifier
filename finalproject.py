# This script trains and saves the model
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import sys

# Fix for Unicode encoding error
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Define image dimensions
img_width = 180
img_height = 180

# Set paths to data directories using raw string literals
data_train_path = r'D:\Fruits_Vegetables\Fruits_Vegetables\train'
data_test_path = r'D:\Fruits_Vegetables\Fruits_Vegetables\test'
data_val_path = r'D:\Fruits_Vegetables\Fruits_Vegetables\validation'

# Load the training dataset
data_train = tf.keras.utils.image_dataset_from_directory(
    data_train_path,
    shuffle=True,
    image_size=(img_width, img_height),
    batch_size=32,
    validation_split=False
)

# Print class names to verify the data is loaded correctly
data_cat = data_train.class_names
print(data_cat)

# Load the validation dataset
data_val = tf.keras.utils.image_dataset_from_directory(
    data_val_path,
    shuffle=True,
    image_size=(img_width, img_height),
    batch_size=32,
    validation_split=False
)

# Load the test dataset
data_test = tf.keras.utils.image_dataset_from_directory(
    data_test_path,
    shuffle=True,
    image_size=(img_width, img_height),
    batch_size=32,
    validation_split=False
)

# Normalize the data
normalization_layer = layers.Rescaling(1./255)

data_train = data_train.map(lambda x, y: (normalization_layer(x), y))
data_val = data_val.map(lambda x, y: (normalization_layer(x), y))
data_test = data_test.map(lambda x, y: (normalization_layer(x), y))

# Build the model
model = keras.Sequential([
    layers.Input(shape=(img_width, img_height, 3)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(data_cat), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    data_train,
    validation_data=data_val,
    epochs=10
)

# Evaluate the model
test_loss, test_acc = model.evaluate(data_test)
print(f'Test accuracy: {test_acc}')

# Plot training history
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Save the model
model.save('my_model.h5')  # Save the entire model to a file
print('Model saved to my_model.h5')
