import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import os
import shutil
import pandas as pd

# Creating subdirectories within the cropped_image directory so that we can extract the class information 
# from the csv and parse it together with the cropped images. This is necessary for the keras 

# defining somoe parameters for the loader:
batch_size = 32
# image size is predetermined through the face_recognition filter
img_height = 200
img_width = 200

# # Read the CSV file containing image filenames and their corresponding class labels
# csv_file = 'train_small.csv'
# data = pd.read_csv(csv_file)

# # Define the main directory where the images will be stored
# new_subdir = 'subdir_small_160'
# os.makedirs(new_subdir, exist_ok=True)

# # Iterate over each row in the CSV file
# for index, row in data.iterrows():
#     image_filename = row['file_name']  # Assuming 'filename' is the column containing image filenames
#     class_label = row['category']  # Assuming 'class' is the column containing class labels
    
#     # Create subdirectory for the class if it doesn't exist
#     class_directory = os.path.join(new_subdir, str(class_label))
#     os.makedirs(class_directory, exist_ok=True)
    
#     # Move or copy the image to the corresponding class subdirectory
#     image_source = 'cropped_small_160/' + image_filename  # Adjust the path as needed
#     if not os.path.exists(image_source):
#         print(f"Image file '{image_filename}' not found. Skipping...")
#         continue
#     image_destination = os.path.join(class_directory, image_filename)
#     shutil.copy(image_source, image_destination)  # Use shutil.move() for moving instead of copying


# Dividing the existing dataset into training and validation sets
subdir_small = 'subdir_small'
train_small_ds = tf.keras.utils.image_dataset_from_directory(
  subdir_small,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_small_ds = tf.keras.utils.image_dataset_from_directory(
  subdir_small,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# storing the class names for extraction later
class_names = train_small_ds.class_names
num_classes = len(class_names)

# Configuring the dataset for cashe. I can get data from disk without having I/O becoming blocking. 
AUTOTUNE = tf.data.AUTOTUNE

train_small_ds = train_small_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_small_ds = val_small_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Standardize the data so that the the RGB values will be from 0 - 1 instead of 0 - 255
normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_small_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

# increasing the accuracy method 1: including a augmented model: 
# creating an augmented model to get rid of overfitting
augmented_model = keras.Sequential(
  [
    # performing data augmentation
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),

    # adding in the convolutional and max pooling layers as usual
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='sigmoid'),
    layers.Dense(num_classes, name="outputs")
  ]
)

augmented_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
augmented_model.summary()

# training the data
epochs = 20
history = augmented_model.fit(
  train_small_ds,
  validation_data=val_small_ds,
  epochs=epochs
)

# saving the pretrained baseline model
augmented_model.save('baseline_model.h5')

# showing the performance using accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()