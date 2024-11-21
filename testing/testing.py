import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance, ImageFilter
import random
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

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
        pil_img = enhancer.enhance(random.uniform(1.5, 1.8)) 
        img = np.array(pil_img) / 255.0
        return img

    def _apply_darken(self, img):
        img = np.array(img)
        img = np.multiply(img, random.uniform(0.2, 0.6)) 
        img = np.clip(img, 0, 255)
        return img

    def _add_noise(self, img):
        noise_factor = 0.15
        img = np.array(img)
        noise = np.random.normal(loc=0.0, scale=noise_factor, size=img.shape)
        img = img + noise
        img = np.clip(img, 0, 1) 
        return img

    def _apply_rotation(self, img, angle=15):
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_img = pil_img.rotate(angle)
        img = np.array(pil_img) / 255.0
        return img

    def _apply_zoom(self, img, zoom_factor=1.5):
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        width, height = pil_img.size
        x = width / 2
        y = height / 2
        pil_img = pil_img.crop((x - width / (2 * zoom_factor), y - height / (2 * zoom_factor), x + width / (2 * zoom_factor), y + height / (2 * zoom_factor)))
        pil_img = pil_img.resize((width, height), Image.LANCZOS)
        img = np.array(pil_img) / 255.0
        return img

    def _apply_contrast(self, img):
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(random.uniform(0.5, 1.5))
        img = np.array(pil_img) / 255.0
        return img

    def _apply_blur(self, img):
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1))
        img = np.array(pil_img) / 255.0
        return img

    def _predict(self, test_images):
        predictions = self.model.predict(test_images)
        predicted_labels = np.argmax(predictions, axis=1)
        return predicted_labels

    def _display_images(self, images, titles, num_images=10):
        plt.figure(figsize=(20, 10))
        num_images = min(num_images, len(images))
        grid_size = int(np.ceil(np.sqrt(num_images)))
        for i in range(num_images):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(images[i])
            plt.title(titles[i], fontsize=10)
            plt.axis('off')
        plt.tight_layout(pad=2.0)
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

        print("\nEvaluating with rotated images:")
        rotated_images = [self._apply_rotation(img) for img in test_images]
        rotated_preds = self._predict(np.array(rotated_images))
        rotated_f1 = self._calculate_f1_score(true_labels, rotated_preds, "Rotated")
        results['Rotated'] = rotated_f1

        print("\nEvaluating with zoomed images:")
        zoomed_images = [self._apply_zoom(img) for img in test_images]
        zoomed_preds = self._predict(np.array(zoomed_images))
        zoomed_f1 = self._calculate_f1_score(true_labels, zoomed_preds, "Zoomed")
        results['Zoomed'] = zoomed_f1

        print("\nEvaluating with contrast adjusted images:")
        contrast_images = [self._apply_contrast(img) for img in test_images]
        contrast_preds = self._predict(np.array(contrast_images))
        contrast_f1 = self._calculate_f1_score(true_labels, contrast_preds, "Contrast")
        results['Contrast'] = contrast_f1

        print("\nEvaluating with blurred images:")
        blurred_images = [self._apply_blur(img) for img in test_images]
        blurred_preds = self._predict(np.array(blurred_images))
        blurred_f1 = self._calculate_f1_score(true_labels, blurred_preds, "Blurred")
        results['Blurred'] = blurred_f1

        average_f1 = np.mean(list(results.values()))
        results['Average'] = average_f1

        return results, test_images, test_image_names

    def _calculate_f1_score(self, true_labels, predicted_labels, variation_name):
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        print(f"F1 Score for {variation_name} images: {f1:.4f}")
        return f1

    def _print_predictions(self, test_image_names, predictions):
        for img_name, pred in zip(test_image_names, predictions):
            class_name = self.labels_df[self.labels_df['ClassId'] == pred]['Name'].values[0]
            print(f"Image: {img_name}, Predicted Class Name: {class_name}, Predicted Label: {pred}")


def evaluate_all_models(models_dir, test_dir, labels_df, img_size=(64, 64)):
    results = {}
    sample_indices = None
    sample_images = None
    sample_image_names = None
    transformed_images = None
    for model_file in os.listdir(models_dir):
        if model_file.endswith('.h5'):
            model_path = os.path.join(models_dir, model_file)
            print(f"\nTesting model: {model_file}")
            tester = TestTrafficSigns(model_path, test_dir, labels_df, img_size)
            model_results, test_images, test_image_names = tester.test_with_variations()
            results[model_file] = model_results

            if sample_indices is None:
                sample_indices = random.sample(range(len(test_images)), 1)
                sample_images = [test_images[idx] for idx in sample_indices]
                sample_image_names = [test_image_names[idx] for idx in sample_indices]

                transformed_images = {
                    "Original": sample_images[0],
                    "Brightened": tester._apply_brightness(sample_images[0]),
                    "Darkened": tester._apply_darken(sample_images[0]),
                    "Noisy": tester._add_noise(sample_images[0]),
                    "Rotated": tester._apply_rotation(sample_images[0]),
                    "Zoomed": tester._apply_zoom(sample_images[0]),
                    "Contrast": tester._apply_contrast(sample_images[0]),
                    "Blurred": tester._apply_blur(sample_images[0])
                }

    if transformed_images:
        tester._display_images(
            list(transformed_images.values()),
            list(transformed_images.keys()),
            num_images=len(transformed_images)
        )

        sample_preds = {name: tester._predict(np.array([img]))[0] for name, img in transformed_images.items()}
        sample_titles = [f"{name} - Pred: {tester.labels_df['Name'][pred]}" for name, pred in sample_preds.items()]
        tester._display_images(
            list(transformed_images.values()),
            sample_titles,
            num_images=len(transformed_images)
        )
    results_df = pd.DataFrame(results).T
    print("\nResults:")
    print(results_df)

models_dir = 'models'
test_dir = 'data/traffic_Data/TEST'
labels_df = pd.read_csv('./data/traffic_Data/labels.csv')

evaluate_all_models(models_dir, test_dir, labels_df, img_size=(64, 64))