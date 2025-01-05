import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from scipy.ndimage import rotate, gaussian_filter
from skimage.util import random_noise
from skimage.exposure import adjust_gamma, adjust_sigmoid

# Helper methods to load images, calculate F1 score, and print predictions
# No need to change these
def load_and_preprocess_image(img_path, img_size):
    img = load_img(img_path, target_size=img_size)
    img = img_to_array(img) / 255.0
    return img

def predict(model, test_images):
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    return predicted_labels

def display_images(images, title, num_images=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        plt.title(title)
        plt.axis('off')
    plt.show()

def calculate_f1_score(true_labels, predicted_labels, variation_name):
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print(f"F1 Score for {variation_name} images: {f1:.4f}")
    return f1

def print_predictions(test_image_names, predictions, labels_df):
    for img_name, pred in zip(test_image_names, predictions):
        class_id = img_name.split('_')[0]
        class_name = labels_df[labels_df['ClassId'] == int(class_id)]['Name'].values[0]
        print(f"Image: {img_name}, Predicted Class: {class_name}, Predicted Label: {pred}")

# Example modification
# Rotates the image by 20 degrees
def rotate_image(img, angle=20):
    return rotate(img, angle, reshape=False)

# MR1: Adding Noise (Kenya and Sweden)
def add_noise(img):
    return random_noise(img, mode='gaussian', var=0.02)

# MR2: Blurring Images (Kenya)
def blur_image(img):
    return gaussian_filter(img, sigma=1)

# MR3: Brightening Images (California)
def brighten_image(img):
    return adjust_gamma(img, gamma=0.5)

# MR4: Increasing Saturation (California)
def increase_saturation(img):
    img = adjust_sigmoid(img, cutoff=0.5, gain=10)
    return img

def test_with_variations(model_path, test_dir, labels_df, img_size=(64, 64)):
    model = load_model(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    test_images = []
    test_image_names = []
    true_labels = []
    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        img = load_and_preprocess_image(img_path, img_size)
        test_images.append(img)
        test_image_names.append(img_name)
        class_id = int(img_name.split('_')[0])
        true_labels.append(class_id)

    test_images = np.array(test_images)
    true_labels = np.array(true_labels)

    f1_scores = {} # Store F1 scores for each variation, used for the table later

    print("Evaluating on original images:")
    original_preds = predict(model, test_images)
    original_f1 = calculate_f1_score(true_labels, original_preds, "Original")
    f1_scores["Original"] = original_f1
    display_images(test_images, "Original")

    print("\nEvaluating with rotated images:")
    rotated_images = [rotate_image(img) for img in test_images] # Rotate all the images in the test set
    rotated_preds = predict(model, np.array(rotated_images)) # Predict on the rotated images
    rotated_f1 = calculate_f1_score(true_labels, rotated_preds, "Rotated") # Calculate the F1 score on the rotated images
    f1_scores["Rotated"] = rotated_f1 # Store the F1 score for the rotated images in the dictionary
    display_images(rotated_images, "Rotated") # Display the rotated images

    print("\nEvaluating with noise (MR1):")
    noisy_images = [add_noise(img) for img in test_images]
    noisy_preds = predict(model, np.array(noisy_images))
    noisy_f1 = calculate_f1_score(true_labels, noisy_preds, "Noisy")
    f1_scores["Noisy"] = noisy_f1
    display_images(noisy_images, "Noisy")

    print("\nEvaluating with blurred images (MR2):")
    blurred_images = [blur_image(img) for img in test_images]
    blurred_preds = predict(model, np.array(blurred_images))
    blurred_f1 = calculate_f1_score(true_labels, blurred_preds, "Blurred")
    f1_scores["Blurred"] = blurred_f1
    display_images(blurred_images, "Blurred")

    print("\nEvaluating with brightened images (MR3):")
    brightened_images = [brighten_image(img) for img in test_images]
    brightened_preds = predict(model, np.array(brightened_images))
    brightened_f1 = calculate_f1_score(true_labels, brightened_preds, "Brightened")
    f1_scores["Brightened"] = brightened_f1
    display_images(brightened_images, "Brightened")

    print("\nEvaluating with increased saturation (MR4):")
    saturated_images = [increase_saturation(img) for img in test_images]
    saturated_preds = predict(model, np.array(saturated_images))
    saturated_f1 = calculate_f1_score(true_labels, saturated_preds, "Increased Saturation")
    f1_scores["Increased Saturation"] = saturated_f1
    display_images(saturated_images, "Increased Saturation")

    # Prints out the F1 score for the original and modified images
    # Shows the drop of F1 score from the original images in a table
    print("\nF1 Score Table:")
    print(f"{'Variation':<20} {'F1 Score':<10} {'Drop from Original':<20}")
    for variation, f1_score in f1_scores.items():
        drop = original_f1 - f1_score
        print(f"{variation:<20} {f1_score:<10.4f} {drop:<20.4f}")

model_path = "models/modelA'.h5" # Change this to the model you want to test

test_with_variations(
    model_path=model_path,
    test_dir='data/traffic_Data/TEST',
    labels_df=pd.read_csv('data/traffic_Data/labels.csv'),
    img_size=(64, 64) # Do not change this, since the model was trained with this size and won't work with different sizes
)
