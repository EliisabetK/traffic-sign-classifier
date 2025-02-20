# Trains Model D

import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageFilter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data_dir = 'data/traffic_Data/DATA'
labels_csv = 'data/traffic_Data/labels.csv'
test_dir = 'data/traffic_Data/TEST'
model_save_path = 'models/noise.h5'
img_size = (64, 64)
labels_df = pd.read_csv(labels_csv)

def process_img(img_path, noise_p=0.8, blur_p=0.6, dark_p=0.1, dark_f=0.08):
    img = cv2.imread(img_path)
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img_size)
    img = np.stack((cv2.equalizeHist(img) / 255.0,) * 3, axis=-1)
    if np.random.rand() < noise_p:
        img = np.clip(img + np.random.normal(0, np.random.uniform(0.05, 0.25), img.shape), 0, 1)
    if np.random.rand() < blur_p:
        img = np.array(Image.fromarray((img * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1))) / 255.0
    if np.random.rand() < dark_p:
        img = np.clip(img * dark_f, 0, 1)
    return img

def load_data(directory):
    imgs, lbls = [], []
    for lbl_id in labels_df['ClassId']:
        folder = os.path.join(directory, str(lbl_id))
        for img_name in os.listdir(folder):
            imgs.append(process_img(os.path.join(folder, img_name)))
            lbls.append(lbl_id)
    return np.array(imgs), to_categorical(np.array(lbls))

def load_test(directory):
    imgs, lbls = [], []
    for img_name in os.listdir(directory):
        imgs.append(process_img(os.path.join(directory, img_name)))
        lbls.append(int(img_name.split('_')[0]))
    return np.array(imgs), np.array(lbls)

X, y = load_data(data_dir)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, shear_range=0.2, rotation_range=20)
datagen.fit(X_train)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(labels_df), activation='softmax')
])

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(datagen.flow(X_train, y_train, batch_size=32), validation_data=(X_val, y_val), epochs=80)
val_acc = model.evaluate(X_val, y_val)[1]
print(f"Validation Accuracy: {val_acc*100:.2f}%")
model.save(model_save_path)

X_test, y_test = load_test(test_dir)
preds = np.argmax(model.predict(X_test), axis=1)
print(f"Test Accuracy: {np.mean(preds == y_test)*100:.2f}%")
