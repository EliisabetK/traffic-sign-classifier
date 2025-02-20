import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# File for testing the models with different variations on the test data
# Mainly for development purposes

def load_image(img_path, img_size=(64, 64)):
    img = load_img(img_path, target_size=img_size)
    img = img_to_array(img) / 255.0
    return img

def brighten(img, factor=1.5):
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(factor)
    img = np.array(pil_img) / 255.0
    return img

def extreme_bright(img):
    return brighten(img, factor=2.5)

def gradual_bright(img):
    return brighten(img, factor=random.uniform(1.1, 2.0))

def darken(img):
    img = np.array(img) * 0.5
    return np.clip(img, 0, 1)

def noise(img, noise_level=0.15):
    noise = np.random.normal(0, noise_level, img.shape)
    img = img + noise
    return np.clip(img, 0, 1)

def rotate(img, angle=15):
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    pil_img = pil_img.rotate(angle)
    return np.array(pil_img) / 255.0

def zoom(img):
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    width, height = pil_img.size
    x = width / 2
    y = height / 2
    pil_img = pil_img.crop((x - width / 3, y - height / 3, x + width / 3, y + height / 3))
    pil_img = pil_img.resize((width, height), Image.LANCZOS)
    return np.array(pil_img) / 255.0

def contrast(img, factor=1.5):
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(factor)
    return np.array(pil_img) / 255.0

def saturate(img, factor=2.0):
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    enhancer = ImageEnhance.Color(pil_img)
    pil_img = enhancer.enhance(factor)
    return np.array(pil_img) / 255.0

def blur(img):
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1))
    return np.array(pil_img) / 255.0

def predict(model, test_images):
    predictions = model.predict(test_images)
    return np.argmax(predictions, axis=1)

def calculate_f1(true_labels, predicted_labels, variation_name):
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print(f"F1 Score for {variation_name} images: {f1:.4f}")
    return f1

def evaluate(models_dir, test_dir, labels_df, img_size=(64, 64)):
    results = {}
    for model_file in os.listdir(models_dir):
        if model_file.endswith('.h5'):
            model_path = os.path.join(models_dir, model_file)
            print(f"\nTesting model: {model_file}")
            model = load_model(model_path)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            test_images = []
            true_labels = []
            for img_name in os.listdir(test_dir):
                img_path = os.path.join(test_dir, img_name)
                img = load_image(img_path, img_size)
                test_images.append(img)
                class_id = int(img_name.split('_')[0])
                true_labels.append(class_id)

            test_images = np.array(test_images)
            true_labels = np.array(true_labels)
            variations = {
                "Original": test_images,
                "Brightened": [brighten(img) for img in test_images],
                "Extreme Brightness": [extreme_bright(img) for img in test_images],
                "Gradual Brightness": [gradual_bright(img) for img in test_images],
                "Darkened": [darken(img) for img in test_images],
                "Noise": [noise(img) for img in test_images],
                "Rotated": [rotate(img) for img in test_images],
                "Zoomed": [zoom(img) for img in test_images],
                "High Contrast": [contrast(img, factor=2.0) for img in test_images],
                "Blurred": [blur(img) for img in test_images],
                "Saturated": [saturate(img) for img in test_images]
            }

            results[model_file] = {}
            for variation_name, var_images in variations.items():
                var_images = np.array(var_images)
                print(f"\nEvaluating with {variation_name} images:")
                preds = predict(model, var_images)
                f1 = calculate_f1(true_labels, preds, variation_name)
                results[model_file][f"{variation_name}_F1"] = f1
            results[model_file]["Average_F1"] = np.mean(list(results[model_file].values()))

    results_df = pd.DataFrame(results).T
    print("\nResults:")
    print(results_df)

models_dir = 'models'
test_dir = 'data/traffic_Data/TEST'
labels_df = pd.read_csv('./data/traffic_Data/labels.csv')

evaluate(models_dir, test_dir, labels_df, img_size=(64, 64))
