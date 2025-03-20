import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from testing.MT_testing import metamorphic_test
from PIL import Image, ImageFilter

data_dir = 'data/traffic_Data/DATA'
test_dir = 'data/traffic_Data/TEST'
labels_csv = 'data/traffic_Data/labels.csv'
output_dir = 'models'
img_size = (64, 64)
labels_df = pd.read_csv(labels_csv)

def process_img(img_path, noise_p=0.05, blur_p=0.05, bright_p=0.8, bright_f=2, sat_p=0.8, sat_f_range=(0.7, 2)):
    img = cv2.imread(img_path)
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img_size)
    img = np.stack((cv2.equalizeHist(img) / 255.0,) * 3, axis=-1)

    if np.random.rand() < noise_p:
        img = np.clip(img + np.random.normal(0, np.random.uniform(0.05, 0.20), img.shape), 0, 1)

    if np.random.rand() < blur_p:
        img = np.array(Image.fromarray((img * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1))) / 255.0

    if np.random.rand() < bright_p:
        random_bright_factor = np.random.uniform(0.8, bright_f)
        img = np.clip(img * random_bright_factor, 0, 1)

    if np.random.rand() < sat_p:
        sat_f = np.random.uniform(*sat_f_range)
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * sat_f, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB) / 255.0
    return img

def load_data(directory):
    images, labels = [], []
    for label_id in labels_df['ClassId']:
        folder = os.path.join(directory, str(label_id))
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            images.append(process_img(img_path))
            labels.append(label_id)
    return np.array(images), to_categorical(np.array(labels))

def load_test_data(directory):
    images, labels = [], []
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        images.append(process_img(img_path))
        labels.append(int(img_name.split('_')[0]))
    return np.array(images), np.array(labels)

def build_model():
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
        Dense(len(labels_df), activation='softmax'),
    ])
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

X, y = load_data(data_dir)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

test_images, test_labels = load_test_data(test_dir)
datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, shear_range=0.2, rotation_range=20)

for i in range(25):
    model = build_model()
    model.fit(datagen.flow(X_train, y_train, batch_size=32), validation_data=(X_val, y_val), epochs=100, verbose=0, callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
    temp_path = f"{output_dir}/temp_model_{i}.h5"
    model.save(temp_path)
    results = metamorphic_test(temp_path, test_images, [str(j) for j in range(len(test_images))])

    brightness_consistency = results[4]
    saturation_consistency = results[5]
    darkness_consistency = results[6]
    noise_consistency = results[3]
    blur_consistency = results[2]

    if 92 <= brightness_consistency <= 100 and 92 <= saturation_consistency <= 100 and all(val < 85 for val in [darkness_consistency, noise_consistency, blur_consistency]):
        os.rename(temp_path, f"{output_dir}/bright.h5")
        print("Saved bright.h5")
        break
else:
    print("No model met the criteria")