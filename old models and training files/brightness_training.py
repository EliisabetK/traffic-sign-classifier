import os
import random
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import EarlyStopping
import cv2
import itertools
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

data_dir = 'data/traffic_Data/DATA'
labels_csv = 'data/traffic_Data/labels.csv'
test_dir = 'data/traffic_Data/TEST'
model_save_path = 'models/best_model_with_brightening.h5'
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

def brighten_image(image, factor=1.7):
    return np.clip(image * factor, 0, 1)

def adjust_contrast(image, factor=1.5):
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    return np.clip((image - mean) * factor + mean, 0, 1)

def add_yellow_tint(image, intensity=0.2):
    yellow_tint = np.array([1.0, 1.0, 0.0]) * intensity
    return np.clip(image + yellow_tint, 0, 1)

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
labels = np.array(labels)

labels = to_categorical(labels)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

def visualize_transformations(images, num_images=5, brighten_factor=1.7, contrast_factor=1.4, yellow_intensity=0.2):
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        original_img = images[i]
        brightened_img = brighten_image(original_img, brighten_factor)
        contrast_img = adjust_contrast(brightened_img, contrast_factor)
        yellow_tinted_img = add_yellow_tint(contrast_img, yellow_intensity)

        # Display original image
        plt.subplot(3, num_images, i + 1)
        plt.imshow(original_img)
        plt.title("Original")
        plt.axis('off')

        # Display brightened image
        plt.subplot(3, num_images, num_images + i + 1)
        plt.imshow(brightened_img)
        plt.title("Brightened")
        plt.axis('off')

        # Display yellow tinted image
        plt.subplot(3, num_images, 2 * num_images + i + 1)
        plt.imshow(yellow_tinted_img)
        plt.title("Yellow Tinted")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Select a few images to visualize
sample_images = images[:5]  # Assuming 'images' is your dataset
visualize_transformations(sample_images)

def augment_with_random_transformations(generator, X, y, batch_size=32, brighten_factor=1.5, contrast_factor=1.2, yellow_intensity=0.2, augmentation_probability=0.6):
    while True:
        for X_batch, y_batch in generator.flow(X, y, batch_size=batch_size):
            augmented_X_batch = []
            for img in X_batch:
                if random.random() < augmentation_probability:
                    img = brighten_image(img, brighten_factor)
                    img = adjust_contrast(img, contrast_factor)
                    img = add_yellow_tint(img, yellow_intensity)
                augmented_X_batch.append(img)
            augmented_X_batch = np.array(augmented_X_batch)
            yield augmented_X_batch, y_batch

train_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    rotation_range=20.
)

train_generator = augment_with_random_transformations(train_datagen, X_train, y_train)

val_datagen = ImageDataGenerator()
val_generator = augment_with_random_transformations(val_datagen, X_val, y_val)

param_grid = {
    'dense_units': [512, 1024],
    'dropout_rate': [0.4, 0.5],
    'learning_rate': [0.001, 0.0001],
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
        epochs=65,
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
        best_test_model = model  # Assigning the model correctly
        best_test_params = params
        model.save(model_save_path)  # Ensures the best model is saved