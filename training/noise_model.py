import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2

data_dir = 'data/traffic_Data/DATA'
labels_csv = 'data/traffic_Data/labels.csv'
test_dir = 'data/traffic_Data/TEST'
model_save_path = 'models/noise.h5'

img_size = (64, 64)

labels_df = pd.read_csv(labels_csv)

def preprocess_image_with_noise(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    img = np.stack((img,)*3, axis=-1)
    noise = np.random.normal(loc=0.0, scale=0.2, size=img.shape)
    img = img + noise
    img = np.clip(img, 0.0, 1.0)
    return img

images = []
labels = []

for label_id, label_name in zip(labels_df['ClassId'], labels_df['Name']):
    label_folder = os.path.join(data_dir, str(label_id))
    for img_name in os.listdir(label_folder):
        img_path = os.path.join(label_folder, img_name)
        img = preprocess_image_with_noise(img_path)
        images.append(img)
        labels.append(label_id)

images = np.array(images)
labels = np.array(labels)

labels = to_categorical(labels)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

print(f"Training Data Shape: {X_train.shape}, Validation Data Shape: {X_val.shape}")

datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10.
)
datagen.fit(X_train)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(labels_df), activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=50
)

val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")

model.save(model_save_path)
print(f"Model saved to {model_save_path}")

test_images = []
test_labels = []
test_image_names = []

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    img = preprocess_image_with_noise(img_path)
    test_images.append(img)
    test_image_names.append(img_name)
    class_id = int(img_name.split('_')[0])
    test_labels.append(class_id)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

accuracy = np.mean(predicted_labels == test_labels)
print(f"Test Accuracy: {accuracy*100:.2f}%")