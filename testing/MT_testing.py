import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from scipy.ndimage import rotate, gaussian_filter
from PIL import Image, ImageEnhance

def preprocess(img_path, img_size):
    img = load_img(img_path, target_size=img_size)
    img = img_to_array(img) / 255.0
    return img

def predict(model, test_images):
    predictions = model.predict(test_images)
    return np.argmax(predictions, axis=1)

def rotate_image(img, angle=20):
    return rotate(img, angle, reshape=False)
def blur_image(img, sigma=1):
    return gaussian_filter(img, sigma=sigma)

def add_noise(img, noise_factor=0.15):
    return np.clip(img + np.random.normal(0, noise_factor, img.shape), 0, 1)

def brighten_image(img, factor=1.8):
    img = Image.fromarray((img * 255).astype(np.uint8))
    img = ImageEnhance.Brightness(img).enhance(factor)
    return np.array(img) / 255.0

def increase_saturation(img, factor=1.8):
    img = Image.fromarray((img * 255).astype(np.uint8))
    img = ImageEnhance.Color(img).enhance(factor)
    return np.array(img) / 255.0

def darken_image(img, factor=0.5):
    img = Image.fromarray((img * 255).astype(np.uint8))
    img = ImageEnhance.Brightness(img).enhance(factor)
    return np.array(img) / 255.0

def metamorphic_test(model_path, test_images, test_image_names):
    print(f"\nEvaluating model: {model_path}")

    model = load_model(model_path)
    original_preds = predict(model, test_images)

    rotated_preds = predict(model, np.array([rotate_image(img) for img in test_images]))
    blurred_preds = predict(model, np.array([blur_image(img) for img in test_images]))
    noise_preds = predict(model, np.array([add_noise(img) for img in test_images]))
    brightened_preds = predict(model, np.array([brighten_image(img) for img in test_images]))
    saturated_preds = predict(model, np.array([increase_saturation(img) for img in test_images]))
    darkened_preds = predict(model, np.array([darken_image(img) for img in test_images]))

    def consistency(orig, mod):
        return sum(1 for o, m in zip(orig, mod) if o == m) / len(orig) * 100

    return (model_path, 
        consistency(original_preds, rotated_preds),
        consistency(original_preds, blurred_preds),
        consistency(original_preds, noise_preds),
        consistency(original_preds, brightened_preds),
        consistency(original_preds, saturated_preds),
        consistency(original_preds, darkened_preds)
    )

def run(models_dir, test_dir, img_size=(64, 64)):
    test_images = []
    test_image_names = []

    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        img = preprocess(img_path, img_size)
        test_images.append(img)
        test_image_names.append(img_name)

    test_images = np.array(test_images)
    results = []
    for model_file in os.listdir(models_dir):
        if model_file.endswith('.h5'):  
            model_path = os.path.join(models_dir, model_file)
            result = metamorphic_test(model_path, test_images, test_image_names)
            results.append(result)
    results_df = pd.DataFrame(results, columns=[
        "Model name", 
        "Rotation consistency", 
        "Blur consistency",
        "Noise consistency",
        "Brightness consistency",
        "Saturation consistency",
        "Darkness consistency"
    ])
    print("\nMT results:")
    print(results_df.to_string(index=False))
    return results_df

models_dir = "models"
test_dir = "data/traffic_Data/TEST"
results_df = run(models_dir, test_dir)