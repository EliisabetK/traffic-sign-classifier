import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

def load_and_preprocess_image(img_path, img_size=(64, 64)):
    img = load_img(img_path, target_size=img_size)
    img = img_to_array(img) / 255.0
    return img

def apply_brightness(img):
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(1.8)
    img = np.array(pil_img) / 255.0
    return img

def apply_darken(img):
    img = np.array(img)
    img = np.multiply(img, 0.4)
    img = np.clip(img, 0, 255)
    return img

def add_noise(img):
    noise = np.full_like(img, 0.15)
    img = img + noise
    img = np.clip(img, 0, 1)
    return img

def apply_rotation(img):
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    pil_img = pil_img.rotate(15)
    img = np.array(pil_img) / 255.0
    return img

def apply_zoom(img):
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    width, height = pil_img.size
    x = width / 2
    y = height / 2
    pil_img = pil_img.crop((x - width / 3, y - height / 3, x + width / 3, y + height / 3))
    pil_img = pil_img.resize((width, height), Image.LANCZOS)
    img = np.array(pil_img) / 255.0
    return img

def apply_contrast(img):
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.0)
    img = np.array(pil_img) / 255.0
    return img

def apply_blur(img):
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1))
    img = np.array(pil_img) / 255.0
    return img

def apply_occlusion(img):
    np_img = np.array(img)
    h, w, _ = np_img.shape
    np_img[10:12, 10:12] = 0
    img = np.clip(np_img, 0, 255).astype(np.uint8)
    img = np.array(img) / 255.0
    return img

def predict(model, test_images):
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    return predicted_labels

def display_images(images, titles, num_images=10):
    plt.figure(figsize=(20, 10))
    num_images = min(num_images, len(images))
    grid_size = int(np.ceil(np.sqrt(num_images)))
    for i in range(num_images):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i], fontsize=10)
        plt.axis('off')
    plt.tight_layout(pad=2.0)

def calculate_f1_score(true_labels, predicted_labels, variation_name):
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print(f"F1 Score for {variation_name} images: {f1:.4f}")
    return f1

def evaluate_all_models(models_dir, test_dir, labels_df, img_size=(64, 64)):
    results = {}
    for model_file in os.listdir(models_dir):
        if model_file.endswith('.h5'):
            model_path = os.path.join(models_dir, model_file)
            print(f"\nTesting model: {model_file}")
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
            
            print("Evaluating on original images:")
            original_preds = predict(model, test_images)
            original_f1 = calculate_f1_score(true_labels, original_preds, "Original")
            results[model_file] = {"Original_F1": original_f1}

            print("\nEvaluating with increased brightness:")
            brightened_images = [apply_brightness(img) for img in test_images]
            brightened_preds = predict(model, np.array(brightened_images))
            brightened_f1 = calculate_f1_score(true_labels, brightened_preds, "Brightened")
            results[model_file]["Brightened_F1"] = brightened_f1

            print("\nEvaluating with darkened images:")
            darkened_images = [apply_darken(img) for img in test_images]
            darkened_preds = predict(model, np.array(darkened_images))
            darkened_f1 = calculate_f1_score(true_labels, darkened_preds, "Darkened")
            results[model_file]["Darkened_F1"] = darkened_f1

            print("\nEvaluating with noisy images:")
            noisy_images = [add_noise(img) for img in test_images]
            noisy_preds = predict(model, np.array(noisy_images))
            noisy_f1 = calculate_f1_score(true_labels, noisy_preds, "Noisy")
            results[model_file]["Noisy_F1"] = noisy_f1

            print("\nEvaluating with rotated images:")
            rotated_images = [apply_rotation(img) for img in test_images]
            rotated_preds = predict(model, np.array(rotated_images))
            rotated_f1 = calculate_f1_score(true_labels, rotated_preds, "Rotated")
            results[model_file]["Rotated_F1"] = rotated_f1

            print("\nEvaluating with zoomed images:")
            zoomed_images = [apply_zoom(img) for img in test_images]
            zoomed_preds = predict(model, np.array(zoomed_images))
            zoomed_f1 = calculate_f1_score(true_labels, zoomed_preds, "Zoomed")
            results[model_file]["Zoomed_F1"] = zoomed_f1

            print("\nEvaluating with contrast adjusted images:")
            contrast_images = [apply_contrast(img) for img in test_images]
            contrast_preds = predict(model, np.array(contrast_images))
            contrast_f1 = calculate_f1_score(true_labels, contrast_preds, "Contrast")
            results[model_file]["Contrast_F1"] = contrast_f1

            print("\nEvaluating with blurred images:")
            blurred_images = [apply_blur(img) for img in test_images]
            blurred_preds = predict(model, np.array(blurred_images))
            blurred_f1 = calculate_f1_score(true_labels, blurred_preds, "Blurred")
            results[model_file]["Blurred_F1"] = blurred_f1

            print("\nEvaluating with occluded images:")
            occluded_images = [apply_occlusion(img) for img in test_images]
            occluded_preds = predict(model, np.array(occluded_images))
            occluded_f1 = calculate_f1_score(true_labels, occluded_preds, "Occluded")
            results[model_file]["Occluded_F1"] = occluded_f1

            results[model_file]["Average_F1"] = np.mean(list(results[model_file].values()))

    results_df = pd.DataFrame(results).T
    print("\nResults:")
    print(results_df)

models_dir = 'models'
test_dir = 'data/traffic_Data/TEST'
labels_df = pd.read_csv('./data/traffic_Data/labels.csv')

evaluate_all_models(models_dir, test_dir, labels_df, img_size=(64, 64))
