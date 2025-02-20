# Trains Model A

import os
import numpy as np
import pandas as pd
import cv2
import random
import itertools
from PIL import Image, ImageEnhance
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_dir = 'data/traffic_Data/DATA'
labels_csv = 'data/traffic_Data/labels.csv'
test_dir = 'data/traffic_Data/TEST'
model_save_path = 'models/brightness.h5'
img_size = (64, 64)
labels_df = pd.read_csv(labels_csv)

def process_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img_size)
    img = np.stack((cv2.equalizeHist(img) / 255.0,) * 3, axis=-1)
    return img

def adjust_brightness(img, factor):
    img = Image.fromarray((img * 255).astype(np.uint8))
    img = ImageEnhance.Brightness(img).enhance(factor)
    return np.array(img) / 255.0

def adjust_saturation(img, factor):
    img = Image.fromarray((img * 255).astype(np.uint8))
    img = ImageEnhance.Color(img).enhance(factor)
    return np.array(img) / 255.0

def add_noise(img, noise_factor=0.05):
    return np.clip(img + np.random.normal(0, noise_factor, img.shape), 0, 1)

def augment_data(generator, X, y, batch_size=32):
    while True:
        for X_batch, y_batch in generator.flow(X, y, batch_size=batch_size):
            dark_X = np.array([adjust_brightness(img, random.uniform(1, 2.0)) for img in X_batch])
            sat_X = np.array([adjust_saturation(img, random.uniform(1, 2.0)) for img in X_batch])
            noisy_X = np.array([add_noise(img, 0.15) for img in X_batch])
            X_aug = np.concatenate([X_batch, dark_X, sat_X, noisy_X])
            y_aug = np.concatenate([y_batch, y_batch, y_batch, y_batch])
            yield X_aug, y_aug

X, y = zip(*[(process_img(os.path.join(data_dir, str(lbl_id), img)), lbl_id) 
              for lbl_id in labels_df['ClassId'] for img in os.listdir(os.path.join(data_dir, str(lbl_id)))])
X, y = np.array(X), to_categorical(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, shear_range=0.2, rotation_range=20)
train_gen = augment_data(datagen, X_train, y_train)
val_gen = augment_data(ImageDataGenerator(), X_val, y_val)

params = list(itertools.product([512], [0.5], [0.001]))
best_model, best_acc, best_f1, best_params = None, 0, 0, None

X_test, y_test = zip(*[(process_img(os.path.join(test_dir, img)), int(img.split('_')[0])) for img in os.listdir(test_dir)])
X_test, y_test = np.array(X_test), np.array(y_test)

for dense_units, dropout_rate, lr in params:
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
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(dense_units // 2, activation='relu'),
        Dropout(dropout_rate),
        Dense(len(labels_df), activation='softmax'),
    ])
    model.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, steps_per_epoch=len(X_train) // 32, validation_data=val_gen, validation_steps=len(X_val) // 32, epochs=90, callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)], verbose=0)
    y_pred = np.argmax(model.predict(X_val), axis=1)
    val_f1 = f1_score(np.argmax(y_val, axis=1), y_pred, average='weighted')
    test_preds = np.argmax(model.predict(X_test), axis=1)
    test_acc, test_f1 = np.mean(test_preds == y_test), f1_score(y_test, test_preds, average='weighted')
    if test_acc > best_acc or (test_acc == best_acc and test_f1 > best_f1):
        best_acc, best_f1, best_model, best_params = test_acc, test_f1, model, (dense_units, dropout_rate, lr)

if best_model:
    best_model.save(model_save_path)
    print(f"Best Model saved to {model_save_path} with test accuracy {best_acc*100:.2f}% and test F1 score {best_f1:.2f}")
    print(f"Best Parameters: {best_params}")
