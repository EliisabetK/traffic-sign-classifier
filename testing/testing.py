import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance, ImageFilter
import random
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

def load_and_preprocess_image(img_path, img_size=(64, 64)):
    img = load_img(img_path, target_size=img_size)
    img = img_to_array(img) / 255.0
    return img

def apply_brightness(img):
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(random.uniform(1.5, 1.8))
    img = np.array(pil_img) / 255.0
    return img

def apply_darken(img):
    img = np.array(img)
    img = np.multiply(img, random.uniform(0.2, 0.6))
    img = np.clip(img, 0, 255)
    return img

def add_noise(img):
    noise_factor = 0.15
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=img.shape)
    img = img + noise
    img = np.clip(img, 0, 1)
    return img

def apply_rotation(img, angle=15):
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    pil_img = pil_img.rotate(angle)
    img = np.array(pil_img) / 255.0
    return img

def apply_zoom(img, zoom_factor=1.5):
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    width, height = pil_img.size
    x = width / 2
    y = height / 2
    pil_img = pil_img.crop((x - width / (2 * zoom_factor), y - height / (2 * zoom_factor), x + width / (2 * zoom_factor), y + height / (2 * zoom_factor)))
    pil_img = pil_img.resize((width, height), Image.LANCZOS)
    img = np.array(pil_img) / 255.0
    return img

def apply_contrast(img):
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(random.uniform(0.5, 1.5))
    img = np.array(pil_img) / 255.0
    return img

def apply_blur(img):
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1))
    img = np.array(pil_img) / 255.0
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
    plt.show()

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

            results[model_file]["Average_F1"] = np.mean(list(results[model_file].values()))

            sample_idx = random.randint(0, len(test_images) - 1)
            sample_image = test_images[sample_idx]
            sample_true_label = true_labels[sample_idx]
            sample_image_name = test_image_names[sample_idx]
            
            transformed_images = {
                "Original": sample_image,
                "Brightened": apply_brightness(sample_image),
                "Darkened": apply_darken(sample_image),
                "Noisy": add_noise(sample_image),
                "Rotated": apply_rotation(sample_image),
                "Zoomed": apply_zoom(sample_image),
                "Contrast": apply_contrast(sample_image),
                "Blurred": apply_blur(sample_image),
            }
            
            transformed_preds = {name: predict(model, np.array([img]))[0] for name, img in transformed_images.items()}
            visualize(
                transformed_images, transformed_preds, sample_true_label, sample_image_name, labels_df, model_file
            )

    results_df = pd.DataFrame(results).T
    print("\nResults:")
    print(results_df)

def visualize(transformed_images, transformed_preds, true_label, image_name, labels_df, model_name):
    """Display one image across all transformations with predictions."""
    plt.figure(figsize=(20, 10))
    num_transformations = len(transformed_images)
    grid_size = int(np.ceil(np.sqrt(num_transformations)))
    
    true_label_name = labels_df.loc[labels_df['ClassId'] == true_label, 'Name'].values[0]
    
    for i, (transformation, img) in enumerate(transformed_images.items()):
        pred_label = transformed_preds[transformation]
        pred_label_name = labels_df.loc[labels_df['ClassId'] == pred_label, 'Name'].values[0]
        
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(img)
        plt.title(f"{transformation}\nTrue: {true_label_name}\nPred: {pred_label_name}", fontsize=10)
        plt.axis('off')
    
    plt.suptitle(f"Model: {model_name} - Transformations for {image_name}", fontsize=16)
    plt.tight_layout(pad=3.0)
    plt.show()

models_dir = 'models'
test_dir = 'data/traffic_Data/TEST'
labels_df = pd.read_csv('./data/traffic_Data/labels.csv')

evaluate_all_models(models_dir, test_dir, labels_df, img_size=(64, 64))
