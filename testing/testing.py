import os
import numpy as np
import pandas as pd
import MT_testing as mt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

def load_image(img_path, img_size=(64, 64)):
    try:
        img = load_img(img_path, target_size=img_size)
        img = img_to_array(img) / 255.0
        return img
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

def predict(model, test_images):
    if len(test_images) == 0:
        return np.array([])
    predictions = model.predict(test_images, verbose=0)
    return np.argmax(predictions, axis=1)

def evaluate(models_dir, test_dir, img_size=(64, 64)):
    if not os.path.exists(models_dir):
        return  
    test_images, true_labels, test_image_names = [], [], []
    print("Loading test images...")
    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        img = load_image(img_path, img_size)
        if img is not None:
            test_images.append(img)
            test_image_names.append(img_name)
            try:
                true_labels.append(int(img_name.split('_')[0]))
            except ValueError:
                print(f"Skip {img_name}")
    
    if not test_images:
        print("Error: No valid test images found.")
        return
    
    test_images = np.array(test_images)
    true_labels = np.array(true_labels)
    print(f"Loaded {len(test_images)} test images.")
    for model_file in os.listdir(models_dir):
        if model_file.endswith('.h5'):
            model_path = os.path.join(models_dir, model_file)
            print(f"\nEvaluating Model: {model_file}")
            try:
                model = load_model(model_path)
                print(f"Model {model_file} loaded successfully")  
                original_preds = predict(model, test_images)

                transformations = {
                    "Rotated": mt.rotate_image,
                    "Blurred": mt.blur_image,
                    "Noisy": mt.add_noise,
                    "Brightened": mt.brighten_image,
                    "Saturated": mt.increase_saturation,
                    "Darkened": mt.darken_image,
                }
                transformed_preds = {}
                for transform_name, transform_func in transformations.items():
                    transformed_images = np.array([transform_func(img) for img in test_images])
                    transformed_preds[transform_name] = predict(model, transformed_images)                
                model_name = model_file.replace(".h5", "")
                csv_path = os.path.join(models_dir, f"detailed_predictions_{model_name}.csv")
                detailed_predictions = [
                    [test_image_names[i], true_labels[i], original_preds[i]] +
                    [transformed_preds[trans][i] for trans in transformations.keys()]
                    for i in range(len(test_image_names))
                ]
                
                predictions_df = pd.DataFrame(detailed_predictions, columns=[
                    "Image", "True label", "Original pred"] + list(transformations.keys())
                )
                
                print(f"Saving CSV to: {csv_path}")
                try:
                    predictions_df.to_csv(csv_path, index=False)
                    print(f"CSV saved successfully: {csv_path}")
                except Exception as e:
                    print(f"Error saving CSV: {e}")
            except Exception as e:
                print(f"Error processing {model_file}: {e}")

models_dir = "models"
test_dir = "data/traffic_Data/TEST"
evaluate(models_dir, test_dir)
