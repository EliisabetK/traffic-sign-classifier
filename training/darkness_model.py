# Trains Model B

import itertools
import os
import pandas as pd
import numpy as np
import random
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import EarlyStopping
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

data_dir = 'data/traffic_Data/DATA'
labels_csv = 'data/traffic_Data/labels.csv'
test_dir = 'data/traffic_Data/TEST'
model_save_path = 'models/best_model_with_darkening.h5'
img_size = (64, 64)

labels_df = pd.read_csv(labels_csv)

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    return np.stack((img / 255.0,) * 3, axis=-1)

def augment_image(image, darken_factor=0.5, noise_factor=0.1):
    darkened = np.clip(image * random.uniform(0.2, darken_factor), 0, 1)
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=image.shape)
    return np.clip(image + noise, 0, 1), darkened

def load_data(data_dir):
    images, labels = [], []
    for label_id, label_name in zip(labels_df['ClassId'], labels_df['Name']):
        label_folder = os.path.join(data_dir, str(label_id))
        for img_name in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_name)
            img = preprocess_image(img_path)
            images.append(img)
            labels.append(label_id)
    return np.array(images), to_categorical(labels)

images, labels = load_data(data_dir)
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

def augment_with_noise_and_darkening(generator, X, y, batch_size=32):
    while True:
        for X_batch, y_batch in generator.flow(X, y, batch_size=batch_size):
            augmented_X, darkened_X = zip(*(augment_image(img) for img in X_batch))
            yield np.concatenate([X_batch, darkened_X, augmented_X]), np.concatenate([y_batch] * 3)

train_datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, 
                                    zoom_range=0.2, shear_range=0.2, rotation_range=20.)
train_generator = augment_with_noise_and_darkening(train_datagen, X_train, y_train)

val_datagen = ImageDataGenerator()
val_generator = augment_with_noise_and_darkening(val_datagen, X_val, y_val)

def load_test_data(test_dir):
    test_images, test_labels = [], []
    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        img = preprocess_image(img_path)
        test_images.append(img)
        test_labels.append(int(img_name.split('_')[0]))
    return np.array(test_images), np.array(test_labels)

test_images, test_labels = load_test_data(test_dir)

param_grid = {
    'dense_units': [512],
    'dropout_rate': [0.5],
    'learning_rate': [0.001],
}
param_combinations = list(itertools.product(*param_grid.values()))

best_accuracy, best_f1, best_model, best_params = 0, 0, None, None

for dense_units, dropout_rate, learning_rate in param_combinations:
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(dense_units // 2, activation='relu'),
        Dropout(dropout_rate),
        Dense(len(labels_df), activation='softmax'),
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(train_generator, steps_per_epoch=len(X_train) // 32, validation_data=val_generator,
              validation_steps=len(X_val) // 32, epochs=65,
              callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
              verbose=0)

    val_loss, val_accuracy = model.evaluate(val_generator, steps=len(X_val) // 32, verbose=0)
    y_val_pred = np.argmax(model.predict(X_val), axis=1)
    y_val_true = np.argmax(y_val, axis=1)
    val_f1 = f1_score(y_val_true, y_val_pred, average='weighted')

    test_predictions = model.predict(test_images)
    test_pred_labels = np.argmax(test_predictions, axis=1)
    test_accuracy = np.mean(test_pred_labels == test_labels)
    test_f1 = f1_score(test_labels, test_pred_labels, average='weighted')
    
    print(f"Params: {dense_units, dropout_rate, learning_rate}, Test Accuracy: {test_accuracy*100:.2f}%, Test F1 Score: {test_f1:.2f}")

    if test_accuracy > best_accuracy or (test_accuracy == best_accuracy and test_f1 > best_f1):
        best_accuracy, best_f1, best_model, best_params = test_accuracy, test_f1, model, (dense_units, dropout_rate, learning_rate)

if best_model:
    best_model.save(model_save_path)
    print(f"Best Model saved to {model_save_path} with test accuracy {best_accuracy*100:.2f}% and test F1 score {best_f1:.2f}")
    print(f"Best Parameters based on Test Data: {best_params}")
