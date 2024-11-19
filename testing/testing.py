import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from PIL import Image, ImageEnhance
import random
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

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

    def _apply_brightness(self, img):
        pil_img = Image.fromarray((img * 255).astype(np.uint8)) 
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(random.uniform(1.0, 1.7)) 
        img = np.array(pil_img) / 255.0
        return img

    def _apply_darken(self, img):
        img = np.array(img)
        img = np.multiply(img, random.uniform(0.2, 0.6)) 
        img = np.clip(img, 0, 255)
        return img

    def _add_noise(self, img):
        noise_factor = 0.2
        img = np.array(img)
        noise = np.random.normal(loc=0.0, scale=noise_factor, size=img.shape)
        img = img + noise
        img = np.clip(img, 0, 1) 
        return img

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

        for img_name in os.listdir(self.test_dir):
            img_path = os.path.join(self.test_dir, img_name)
            img = self._load_and_preprocess_image(img_path)
            test_images.append(img)
            test_image_names.append(img_name)

            class_id = int(img_name.split('_')[0]) 
            true_labels.append(class_id)

        test_images = np.array(test_images)
        true_labels = np.array(true_labels)

        results = {}

        print("Evaluating on original images:")
        original_preds = self._predict(test_images)
        original_f1 = self._calculate_f1_score(true_labels, original_preds, "Original")
        results['Original'] = original_f1

        print("\nEvaluating with increased brightness:")
        brightened_images = [self._apply_brightness(img) for img in test_images]
        brightened_preds = self._predict(np.array(brightened_images))
        brightened_f1 = self._calculate_f1_score(true_labels, brightened_preds, "Brightened")
        results['Brightened'] = brightened_f1

        print("\nEvaluating with darkened images:")
        darkened_images = [self._apply_darken(img) for img in test_images]
        darkened_preds = self._predict(np.array(darkened_images))
        darkened_f1 = self._calculate_f1_score(true_labels, darkened_preds, "Darkened")
        results['Darkened'] = darkened_f1

        print("\nEvaluating with noisy images:")
        noisy_images = [self._add_noise(img) for img in test_images]
        noisy_preds = self._predict(np.array(noisy_images))
        noisy_f1 = self._calculate_f1_score(true_labels, noisy_preds, "Noisy")
        results['Noisy'] = noisy_f1

        return results

    def _calculate_f1_score(self, true_labels, predicted_labels, variation_name):
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        print(f"F1 Score for {variation_name} images: {f1:.4f}")
        return f1

    def _print_predictions(self, test_image_names, predictions):
        for img_name, pred in zip(test_image_names, predictions):
            class_id = img_name.split('_')[0]
            class_name = self.labels_df[self.labels_df['ClassId'] == int(class_id)]['Name'].values[0]
            print(f"Image: {img_name}, Predicted Class: {class_name}, Predicted Label: {pred}")


def evaluate_all_models(models_dir, test_dir, labels_df, img_size=(64, 64)):
    results = {}
    for model_file in os.listdir(models_dir):
        if model_file.endswith('.h5'):
            model_path = os.path.join(models_dir, model_file)
            print(f"\nTesting model: {model_file}")
            tester = TestTrafficSigns(model_path, test_dir, labels_df, img_size)
            model_results = tester.test_with_variations()
            results[model_file] = model_results
    
    results_df = pd.DataFrame(results).T
    print("\nResults:")
    print(results_df)


models_dir = 'models'
test_dir = 'data/traffic_Data/TEST'
labels_df = pd.read_csv('./data/traffic_Data/labels.csv')

evaluate_all_models(models_dir, test_dir, labels_df, img_size=(64, 64))