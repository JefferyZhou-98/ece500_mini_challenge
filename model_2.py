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
img_height = 160
img_width = 160
IMG_SIZE = (img_height, img_width)

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
subdir_small = 'subdir_small_160'
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


# using Data augmentation
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

# rescaling the pixel values to fit [-1, 1] from [0, 255]
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

# creating baseline model from MobileNet V2
# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.EfficientNetB7(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

# converts 160x160x3 image into 5x5x1280 block of features
image_batch, label_batch = next(iter(train_small_ds))
feature_batch = base_model(image_batch)

# # doing feature extraction
# # freezing the convolutional base
# base_model.trainable = False

# # adding a classification head, averaging over a spatial 5x5 locations, use maxpooling to convert features
# # to a single 1280 elemetn vector per image
# global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# feature_batch_average = global_average_layer(feature_batch)
# # convert all the features into a single prediction per image
# prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
# prediction_batch = prediction_layer(feature_batch_average)

# # building the model
# inputs = tf.keras.Input(shape=(160, 160, 3))
# x = data_augmentation(inputs)
# x = preprocess_input(x)
# x = base_model(x, training=False)
# x = global_average_layer(x)
# x = tf.keras.layers.Dropout(0.2)(x)
# outputs = prediction_layer(x)
# model = tf.keras.Model(inputs, outputs)

# # compiling the model
# base_learning_rate = 0.0001
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
#               loss=tf.keras.losses.BinaryCrossentropy(),
#               metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])

# # training the data
# epochs = 20
# # history = model.fit(
# #   train_small_ds,
# #   validation_data=val_small_ds,
# #   epochs=epochs
# # )

# loss0, accuracy0 = model.evaluate(val_small_ds)
# print("initial loss: {:.2f}".format(loss0))
# print("initial accuracy: {:.2f}".format(accuracy0))

# # saving the pretrained baseline model
# augmented_model.save('baseline_model.h5')

# showing the performance using accuracy and loss
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()