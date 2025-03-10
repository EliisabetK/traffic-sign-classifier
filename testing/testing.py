import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score

def load_image(img_path, img_size=(64, 64)):
    img = load_img(img_path, target_size=img_size)
    img = img_to_array(img) / 255.0
    return img

def predict(model, test_images):
    predictions = model.predict(test_images)
    return np.argmax(predictions, axis=1)

def calculate_f1(true_labels, predicted_labels):
    return f1_score(true_labels, predicted_labels, average='weighted')

def evaluate(models_dir, test_dir, img_size=(64, 64)):
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
            print("\nEvaluating on Original images:")
            preds = predict(model, test_images)
            f1 = calculate_f1(true_labels, preds)
            results[model_file] = {"Original_F1": f1}

    results_df = pd.DataFrame(results).T
    print("\nResults:")
    print(results_df)
    return results_df
models_dir = 'models'
test_dir = 'data/traffic_Data/TEST'
evaluate(models_dir, test_dir, img_size=(64, 64))
