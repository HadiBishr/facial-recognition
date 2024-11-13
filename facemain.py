import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
import os
import cv2
import time

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


#####AUTOMATICALLY TAKING PHOTOS FOR DATASET#####

output_dir = 'Face/1' #Directory where images will be saved
num_images = 8000 # Number of images to capture
interval = 0.01 # Time interval between captures in seconds
alpha = 1.2 # Contrast control (1.0-3.0)
beta = 50 #Brightness control (1-100)

#Initialize the webcam
cap = cv2.VideoCapture(0)

#Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()\


def save_image(frame, img_path):
    # Adjust brightness and contrast
    adjusted_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    # Save the adjusted frame as an image
    cv2.imwrite(img_path, adjusted_frame)


print("Press 's' to start capturing images.")
print("Press 'q' to quit.")

captured_images = 0
executor = ThreadPoolExecutor(max_workers=8)  # Use 4 threads for saving images

while True:
    #Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image.")
        break

    #Display the frame
    cv2.imshow('Frame', frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'): #Start capturing
        start_time = time.time()
        while captured_images < num_images:
            ret, frame = cap.read()

            if not ret:
                print("Failed to capture image.")
                break


            #Save the adjusted frame as an image
            img_path = os.path.join(output_dir, f'image{captured_images + 1}.jpg')
            executor.submit(save_image, frame, img_path)
            captured_images += 1


            #Print status
            if captured_images % 100 == 0:
                print(f'Captured image {captured_images}/{num_images}')

            #Wait for the specified interval
            time.sleep(interval)

        print("Finished capturing images.")
        break

    elif key == ord('q'): #Quit
        break

executor.shutdown(wait=True)

cap.release()
cv2.destroyAllWindows()





dataset_path = "Face"
image_size = (224, 224)
batch_size = 16
seed = 123

from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')


#Loads the full dataset
full_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True,
    seed=seed,
    label_mode='binary',
    color_mode='rgb'
)

# Extract images and labels for stratified splitting
images, labels = [], []
for image_batch, label_batch in full_dataset:
    images.extend(image_batch.numpy())
    labels.extend(label_batch.numpy())

images = np.array(images)
labels = np.array(labels)

images = images / 255.0

train_images, temp_images, train_labels, temp_labels = train_test_split(
    images, labels, test_size=0.4, stratify=labels, random_state=seed)

val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.5, stratify=temp_labels, random_state=seed)


# Convert back to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)





#Data Augmentation
data_augmentation = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomZoom(0.2),
        layers.experimental.preprocessing.RandomContrast(0.2)

    ]
)


#Apply data augmentation
train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

# Prefetching and caching
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)


# Early Stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Learning Rate Scheduling
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=2,
    verbose=1
)



inputs = keras.Input(shape=(224, 224, 3))


x = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)

x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='valid')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.5)(x)  # Add dropout layer

x = layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)

x = layers.Conv2D(filters=512, kernel_size=5, strides=1, padding='valid')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.Flatten()(x)

x = layers.Dense(128)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.5)(x)  # Add dropout layer

x = layers.Dense(32)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

outputs = layers.Dense(1, activation='sigmoid')(x)



model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    loss = keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy'],
)

model.fit(train_dataset, batch_size=batch_size, epochs=100, verbose=2, validation_data=val_dataset, callbacks=[early_stopping, lr_scheduler])
model.evaluate(test_dataset, batch_size=batch_size, verbose=2)

model.save('my_model2.h5')















#
# #Determines the size of the splits
# dataset_size = full_dataset.cardinality().numpy()
# train_size = int(0.6 * dataset_size)
# val_size = int(0.2 * dataset_size)
# test_size = dataset_size - train_size - val_size
#
#
#
# #Splits the dataset
# train_dataset = full_dataset.take(train_size)
# val_test_dataset = full_dataset.skip(train_size) #This takes the rest of the dataset (40 percent) after the training (60 percent). This is temporary
# val_dataset = val_test_dataset.take(val_size)
# test_dataset = val_test_dataset.skip(val_size)
