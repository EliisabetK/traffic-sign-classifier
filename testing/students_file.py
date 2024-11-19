import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from scipy.ndimage import rotate

class TestTrafficSigns:
    def __init__(self, model_path, test_dir, labels_df, img_size=(64, 64)):
        self.model = load_model(model_path)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.test_dir = test_dir
        self.labels_df = labels_df
        self.img_size = img_size

    def _load_and_preprocess_image(self, img_path):
        img = load_img(img_path, target_size=self.img_size)
        img = img_to_array(img) / 255.0
        return img


    def _rotate_image(self, img, angle=45):
        """
        Rotates the image by the given angle.
        """
        return rotate(img, angle, reshape=False)

    '''

    Add more methods as needed for testing the model with different variations of images,

    '''

    def _predict(self, test_images):
        predictions = self.model.predict(test_images)
        predicted_labels = np.argmax(predictions, axis=1)
        return predicted_labels

    def _display_images(self, images, title, num_images=5):
        plt.figure(figsize=(15, 5))
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(images[i])
            plt.title(title)
            plt.axis('off')
        plt.show()

    def test_with_variations(self):
        test_images = []
        test_image_names = []
        true_labels = []

        # Load and preprocess test images
        for img_name in os.listdir(self.test_dir):
            img_path = os.path.join(self.test_dir, img_name)
            img = self._load_and_preprocess_image(img_path)
            test_images.append(img)
            test_image_names.append(img_name)
            class_id = int(img_name.split('_')[0])
            true_labels.append(class_id)

        test_images = np.array(test_images)
        true_labels = np.array(true_labels)

        print("Evaluating on original images:")
        original_preds = self._predict(test_images)
        self._calculate_f1_score(true_labels, original_preds, "Original")
        self._display_images(test_images, "Original")

        # example: Testing with rotated images
        print("\nEvaluating with rotated images:")
        rotated_images = [self._rotate_image(img) for img in test_images]
        rotated_preds = self._predict(np.array(rotated_images))
        self._calculate_f1_score(true_labels, rotated_preds, "Rotated")
        self._display_images(rotated_images, "Rotated")

    def _calculate_f1_score(self, true_labels, predicted_labels, variation_name):
        """
        Calculates and prints the F1 score for a given set of predictions.
        """
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        print(f"F1 Score for {variation_name} images: {f1:.4f}")

    def _print_predictions(self, test_image_names, predictions):
        """
        Prints the predicted and true classes for a subset of images.
        """
        for img_name, pred in zip(test_image_names, predictions):
            class_id = img_name.split('_')[0]
            class_name = self.labels_df[self.labels_df['ClassId'] == int(class_id)]['Name'].values[0]
            print(f"Image: {img_name}, Predicted Class: {class_name}, Predicted Label: {pred}")


model_path = 'traffic_sign_noise_model.h5'

test_traffic_signs = TestTrafficSigns(
    model_path=model_path,
    test_dir='data/traffic_Data/TEST',
    labels_df=pd.read_csv('data/traffic_Data/labels.csv'),
    img_size=(64, 64)
)

test_traffic_signs.test_with_variations()