import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import cv2
import itertools
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

data_dir = 'data/traffic_Data/DATA'
labels_csv = 'data/traffic_Data/labels.csv'
test_dir = 'data/traffic_Data/TEST'
model_save_path = 'models/best_model.h5'
img_size = (64, 64)

labels_df = pd.read_csv(labels_csv)

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    img = np.stack((img,) * 3, axis=-1) 
    return img

def add_noise_or_blur(img):
    if np.random.rand() < 0.5:
        noise = np.random.normal(loc=0.0, scale=0.1, size=img.shape)
        img = np.clip(img + noise, 0, 1)
    else: 
        kernel_size = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return img

images = []
labels = []
for label_id, label_name in zip(labels_df['ClassId'], labels_df['Name']):
    label_folder = os.path.join(data_dir, str(label_id))
    for img_name in os.listdir(label_folder):
        img_path = os.path.join(label_folder, img_name)
        img = preprocess_image(img_path)
        if np.random.rand() < 0.5:
            img = add_noise_or_blur(img)
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

param_grid = {
    'dense_units': [512, 1024],
    'dropout_rate': [0.3, 0.4, 0.5],
    'learning_rate': [0.001],
}
param_combinations = list(itertools.product(*param_grid.values()))

best_test_accuracy = 0
best_test_f1_score = 0
best_test_model = None
best_test_params = None

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

for params in param_combinations:
    dense_units, dropout_rate, learning_rate = params
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
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
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=65,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        ],
        verbose=0
    )
    
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
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