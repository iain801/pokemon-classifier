import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

# Download and explore dataset
import pathlib
dataset_url = "https://github.com/iain801/pokemon-classifier/raw/main/Pokemon.tar"
data_dir = tf.keras.utils.get_file('Pokemon', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# Create a dataset
batch_size = 25
img_height = 400
img_width = 400

# 80/20 validation split
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Print class names
class_names = train_ds.class_names
print(class_names)

# Visualize data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
# Training batches
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Normalize data
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# Create Model
num_classes = 5

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal",input_shape=(img_height,img_width,3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 5, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 5, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 5, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.4),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# Compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# Train Model
epochs = 100
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Visualize training results
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

# Save Model
model.save("pokemon_model")
del model

model = load_model('pokemon_model')

# water Test
water_url = "https://github.com/iain801/pokemon-classifier/raw/main/water-test.jpg"
water_path = tf.keras.utils.get_file('water-test', origin=water_url)

img = keras.preprocessing.image.load_img(
    water_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.title(class_names[np.argmax(score)] + " C=" + str(100 * np.max(score)) + "%")

# fire Test
fire_url = "https://github.com/iain801/pokemon-classifier/raw/main/fire-test.jpg"
fire_path = tf.keras.utils.get_file('fire-test', origin=fire_url)

img = keras.preprocessing.image.load_img(
    fire_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.title(class_names[np.argmax(score)] + " C=" + str(100 * np.max(score)) + "%")

# grass Test
grass_url = "https://github.com/iain801/pokemon-classifier/raw/main/grass-test.jpg"
grass_path = tf.keras.utils.get_file('grass-test', origin=grass_url)

img = keras.preprocessing.image.load_img(
    grass_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.title(class_names[np.argmax(score)] + " C=" + str(100 * np.max(score)) + "%")

# electric Test
electric_url = "https://github.com/iain801/pokemon-classifier/raw/main/electric-test.jpg"
electric_path = tf.keras.utils.get_file('electric-test', origin=electric_url)

img = keras.preprocessing.image.load_img(
    electric_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.title(class_names[np.argmax(score)] + " C=" + str(100 * np.max(score)) + "%")

# poison Test
poison_url = "https://github.com/iain801/pokemon-classifier/raw/main/poison-test.jpg"
poison_path = tf.keras.utils.get_file('poison-test', origin=poison_url)

img = keras.preprocessing.image.load_img(
    poison_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.title(class_names[np.argmax(score)] + " C=" + str(100 * np.max(score)) + "%")