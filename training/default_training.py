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
from PIL import Image, ImageFilter, ImageEnhance

data_dir = 'data/traffic_Data/DATA'
labels_csv = 'data/traffic_Data/labels.csv'
test_dir = 'data/traffic_Data/TEST'
model_save_path = 'models/default.h5'

img_size = (64, 64)

labels_df = pd.read_csv(labels_csv)

def preprocess_image_with_noise_and_blur(img_path, noise_probability=0.05, blur_probability=0.05, darken_probability=0.03, darken_factor=0.8):
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    img = np.stack((img,)*3, axis=-1)
    
    if np.random.rand() < noise_probability:
        noise_factor = np.random.uniform(0.1, 0.2)
        noise = np.random.normal(loc=0.0, scale=noise_factor, size=img.shape)
        img = img + noise
        img = np.clip(img, 0.0, 1.0)
    
    if np.random.rand() < blur_probability:
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1))
        img = np.array(pil_img) / 255.0
    
    if np.random.rand() < darken_probability:
        img = img * darken_factor
        img = np.clip(img, 0.0, 1.0)
    
    return img

images = []
labels = []

for label_id, label_name in zip(labels_df['ClassId'], labels_df['Name']):
    label_folder = os.path.join(data_dir, str(label_id))
    for img_name in os.listdir(label_folder):
        img_path = os.path.join(label_folder, img_name)
        img = preprocess_image_with_noise_and_blur(img_path, noise_probability=0.07, blur_probability=0.05, darken_probability=0.05, darken_factor=0.8) 
        images.append(img)
        labels.append(label_id)

images = np.array(images)
labels = np.array(labels)

labels = to_categorical(labels)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    rotation_range=20.
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

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=85
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
    img = preprocess_image_with_noise_and_blur(img_path, noise_probability=0.05, blur_probability=0.05, darken_probability=0.05, darken_factor=0.7)
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