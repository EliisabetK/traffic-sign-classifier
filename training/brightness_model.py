import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import EarlyStopping
import cv2
import random
from PIL import Image, ImageEnhance, ImageFilter

data_dir = 'data/traffic_Data/DATA'
labels_csv = 'data/traffic_Data/labels.csv'
test_dir = 'data/traffic_Data/TEST'
model_save_path = 'models/br2.h5'
img_size = (64, 64)
num_classes = 58  # Number of classes in the dataset

labels_df = pd.read_csv(labels_csv)

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    img = np.stack((img,) * 3, axis=-1)
    return img

def adjust_brightness(image, factor=1.2):
    return np.clip(image * factor, 0, 1)

def adjust_saturation(image, factor=1.2):
    pil_img = Image.fromarray((image * 255).astype(np.uint8))
    enhancer = ImageEnhance.Color(pil_img)
    pil_img = enhancer.enhance(factor)
    return np.array(pil_img) / 255.0

def add_noise(image, noise_factor=0.05):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=image.shape)
    return np.clip(image + noise, 0, 1)

def apply_blur(image, blur_probability=0.1):
    if random.random() < blur_probability:
        pil_img = Image.fromarray((image * 255).astype(np.uint8))
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1))
        return np.array(pil_img) / 255.0
    return image

images = []
labels = []
for label_id, label_name in zip(labels_df['ClassId'], labels_df['Name']):
    label_folder = os.path.join(data_dir, str(label_id))
    for img_name in os.listdir(label_folder):
        img_path = os.path.join(label_folder, img_name)
        img = preprocess_image(img_path)
        images.append(img)
        labels.append(label_id)

images = np.array(images)
labels = to_categorical(labels)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

def augment_with_random_effects(generator, X, y, batch_size=32):
    while True:
        for X_batch, y_batch in generator.flow(X, y, batch_size=batch_size):
            augmented_X_batch = np.array([
                apply_blur(
                    add_noise(
                        adjust_saturation(
                            adjust_brightness(img, factor=random.uniform(0.8, 1.2)),
                            factor=random.uniform(0.8, 1.2)
                        ),
                        noise_factor=0.05
                    ),
                    blur_probability=0.1
                ) for img in X_batch
            ])
            yield augmented_X_batch, y_batch

train_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    rotation_range=20.
)
train_generator = augment_with_random_effects(train_datagen, X_train, y_train)

val_datagen = ImageDataGenerator()
val_generator = augment_with_random_effects(val_datagen, X_val, y_val)

param_grid = {
    'dense_units': [512],
    'dropout_rate': [0.5],
    'learning_rate': [0.001],
}
param_combinations = list(itertools.product(*param_grid.values()))

test_images = []
test_labels = []
test_image_names = []

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    img = preprocess_image(img_path)
    test_images.append(img)
    test_image_names.append(img_name)
    class_id = int(img_name.split('_')[0])
    test_labels.append(class_id)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

best_test_accuracy = 0
best_test_f1_score = 0
best_test_model = None
best_test_params = None

for params in param_combinations:
    dense_units, dropout_rate, learning_rate = params
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
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(dense_units // 2, activation='relu'),
        Dropout(dropout_rate),
        Dense(len(labels_df), activation='softmax'),
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // 32,
        validation_data=val_generator,
        validation_steps=len(X_val) // 32,
        epochs=75,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=0
    )
    val_loss, val_accuracy = model.evaluate(val_generator, steps=len(X_val) // 32, verbose=0)
    y_val_pred = np.argmax(model.predict(X_val), axis=1)
    y_val_true = np.argmax(y_val, axis=1)
    val_f1 = f1_score(y_val_true, y_val_pred, average='weighted')
    test_predictions = model.predict(test_images)
    test_predicted_labels = np.argmax(test_predictions, axis=1)
    test_accuracy = np.mean(test_predicted_labels == test_labels)
    test_f1 = f1_score(test_labels, test_predicted_labels, average='weighted')
    print(f"Params: {params}, Test Accuracy: {test_accuracy*100:.2f}%, Test F1 Score: {test_f1:.2f}")
    if test_accuracy > best_test_accuracy or (test_accuracy == best_test_accuracy and test_f1 > best_test_f1_score):
        best_test_accuracy = test_accuracy
        best_test_f1_score = test_f1
        best_test_model = model
        best_test_params = params

if best_test_model:
    best_test_model.save(model_save_path)
    print(f"Best Model saved to {model_save_path} with test accuracy {best_test_accuracy*100:.2f}% and test F1 score {best_test_f1_score:.2f}")
    print(f"Best Parameters based on Test Data: {best_test_params}")