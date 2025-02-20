import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from scipy.ndimage import rotate

# Helper methods to load images, calculate F1 score, and print predictions
# No need to change these
def preprocess(img_path, img_size):
    img = load_img(img_path, target_size=img_size)
    img = img_to_array(img) / 255.0
    return img

def predict(model, test_images):
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    return predicted_labels

def display(images, title, num_images=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        plt.title(title)
        plt.axis('off')
    plt.show()

def calculate_f1(true_labels, predicted_labels, variation_name):
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

"""
Add more methods as needed for testing the model with different variations of images
"""

def test(model_path, test_dir, labels_df, img_size=(64, 64)):
    model = load_model(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    test_images = []
    test_image_names = []
    true_labels = []
    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        img = preprocess(img_path, img_size)
        test_images.append(img)
        test_image_names.append(img_name)
        class_id = int(img_name.split('_')[0])
        true_labels.append(class_id)

    test_images = np.array(test_images)
    true_labels = np.array(true_labels)

    f1_scores = {} # Store F1 scores for each variation, used for the table later

    print("Evaluating on original images:")
    original_preds = predict(model, test_images)
    original_f1 = calculate_f1(true_labels, original_preds, "Original")
    f1_scores["Original"] = original_f1
    display(test_images, "Original")

    print("\nEvaluating with rotated images:")
    rotated_images = [rotate_image(img) for img in test_images] # Rotate all the images in the test set
    rotated_preds = predict(model, np.array(rotated_images)) # Predict on the rotated images
    rotated_f1 = calculate_f1(true_labels, rotated_preds, "Rotated") # Calculate the F1 score on the rotated images
    f1_scores["Rotated"] = rotated_f1 # Store the F1 score for the rotated images in the dictionary
    display(rotated_images, "Rotated") # Display the rotated images

    """
    Add more code for the additional variations of images as needed
    """

    # Prints out the F1 score for the original and modified images
    # Shows the drop of F1 score from the original images in a table
    # No need to change this
    print("\nF1 Score Table:")
    print(f"{'Variation':<10} {'F1 Score':<10} {'Drop from Original':<20}")
    for variation, f1_score in f1_scores.items():
        drop = original_f1 - f1_score
        print(f"{variation:<10} {f1_score:<10.4f} {drop:<20.4f}")

model_path = 'models/modelA.h5'

test(
    model_path=model_path,
    test_dir='data/traffic_Data/TEST',
    labels_df=pd.read_csv('data/traffic_Data/labels.csv'),
    img_size=(64, 64) # Do not change this, since the model was trained with this size and won't work with different sizes
)