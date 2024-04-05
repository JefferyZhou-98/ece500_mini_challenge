import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.applications import EfficientNetB0
import pathlib
import os
import shutil
import pandas as pd
import os
import cv2

# Creating subdirectories within the cropped_image directory so that we can extract the class information 
# from the csv and parse it together with the cropped images. This is necessary for the keras 

# defining somoe parameters for the loader:
BATCH_SIZE = 64
# image size is predetermined through the face_recognition filter
img_height = 224
img_width = 224
IMG_SIZE = (img_height, img_width)

# Dividing the existing dataset into training and validation sets
subdir_small = 'subdir_small_224'
train_small_ds = tf.keras.utils.image_dataset_from_directory(
  subdir_small,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=BATCH_SIZE)

val_small_ds = tf.keras.utils.image_dataset_from_directory(
  subdir_small,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=BATCH_SIZE)

# storing the class names for extraction later
class_names = train_small_ds.class_names
num_classes = len(class_names)

# Configuring the dataset for cashe. I can get data from disk without having I/O becoming blocking. 
# AUTOTUNE = tf.data.AUTOTUNE

# train_small_ds = train_small_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# val_small_ds = val_small_ds.cache().prefetch(buffer_size=AUTOTUNE)

# data augmentation
img_augmentation_layers = [
    layers.RandomRotation(factor=0.15),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomFlip(),
    layers.RandomContrast(factor=0.1),
]

def img_augmentation(images):
    for layer in img_augmentation_layers:
        images = layer(images)
    return images

# preparing the inputs. The input data are resized to uniform IMG_SIZE. The labels are input into categorical
# One-hot / categorical encoding
def input_preprocess_train(image, label):
    image = img_augmentation(image)
    label = tf.one_hot(label, 100)
    return image, label


def input_preprocess_test(image, label):
    label = tf.one_hot(label, 100)
    return image, label


train_small_ds = train_small_ds.map(input_preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
# train_small_ds = train_small_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True)
# train_small_ds = train_small_ds.prefetch(tf.data.AUTOTUNE)

val_small_ds = val_small_ds.map(input_preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
# val_small_ds = val_small_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True)


# Training a model using EfficientNetB0
# model = EfficientNetB0(
#     include_top=True,
#     weights=None,
#     classes=100,
#     input_shape=(img_height, img_width, 3),
# )
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# model.summary()

# epochs = 40  # @param {type: "slider", min:10, max:100}
# hist = model.fit(train_small_ds, epochs=epochs, validation_data=val_small_ds)

# transfer learning from pre-trained weights
def build_model(num_classes):
    inputs = layers.Input(shape=(img_height, img_width, 3))
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.01
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
model = build_model(num_classes=100)

def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-30:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )


# training the network
unfreeze_model(model)

# epochs = 5  # @param {type: "slider", min:4, max:10}
# hist = model.fit(train_small_ds, epochs=epochs, validation_data=val_small_ds)
# plot_hist(hist)


# testing dataset 
image_dir = 'test_cropped'
# Generate predictions for the test images
test_files = os.listdir(image_dir)
test_files = [file for file in test_files if file.lower().endswith('.jpg')]
test_files = sorted(test_files, key=lambda x: int(x.split('.')[0]))
# saving into data frame for csv 
data = {'Id': [], 'Category': []}
for file in test_files:
    # loading images
    image_path = os.path.join(image_dir, file)
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)

    # appending to the dataframe
    image_id = file.split('.')[0]
    name = class_names[np.argmax(predictions)]
    data['Id'].append(image_id)
    data['Category'].append(name)
    # print(
    #     "This image most likely belongs to {} with a {:.2f} percent confidence."
    #     .format(class_names[np.argmax(predictions)], 100 * np.max(predictions))
    # )
    # print(image_id)

# Create DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV file
df.to_csv('predictions.csv', index=False)