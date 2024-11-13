import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.utils import array_to_img

import matplotlib.pyplot as plt



dataset_path = "Face"
image_size = (224, 224)
batch_size = 32
seed=123



classOne_folder_path = 'Face/Face1'
save_folder = 'Face/1'




#Data Generators for "1" folder
data_gen = ImageDataGenerator(
    zoom_range=0.2,
    horizontal_flip=False,
    vertical_flip=False,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
)

generator = data_gen.flow_from_directory(
    directory=classOne_folder_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=True,
    subset=None,
    color_mode='rgb'
)


num_batches = 218
for batch_index in range(num_batches):
  #Pulls up the next batch of images from generator
  images = next(generator)

  for i, img in enumerate(images):
    #This converts the image to a format that can be saved
    img = array_to_img(img)

    #This saves the image to the directory specified.
    img.save(os.path.join(save_folder, f'{batch_index * batch_size + i}.jpg'))

  print(f'Batch {batch_index + 1}/{num_batches} processed and saved.')

print('All batches processed and images saved.')








data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
)

train_batches = data_gen.flow_from_directory(
    directory=dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    subset='training',
    color_mode='rgb'
)

validation_batches = data_gen.flow_from_directory(
    directory=dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    color_mode='rgb'
)

test_batches = data_gen.flow_from_directory(
    directory=dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False,
    subset='validation',
    color_mode='rgb'
)




inputs = keras.Input(shape = (224,224, 3))

x = layers.Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)


x = layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'valid')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.MaxPool2D(pool_size = (2,2), padding='valid')(x)

x = layers.Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.Conv2D(filters = 256, kernel_size = 1, strides = 1, padding = 'valid')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.Conv2D(filters = 512, kernel_size = 5, strides = 1, padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)


x = layers.AveragePooling2D(pool_size = (2,2), padding='valid')(x)

x = layers.Conv2D(filters = 1024, kernel_size = 5, strides = 1, padding = 'valid')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)


x = layers.Flatten()(x)

x = layers.Dense(512)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.Dense(256)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.Dense(128)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.Dense(64)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

outputs = layers.Dense(1, activation='sigmoid')(x)