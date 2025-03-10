import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

def preprocess(img_path, img_size):
    img = load_img(img_path, target_size=img_size)
    img = img_to_array(img) / 255.0
    return img

def predict(model, test_images):
    predictions = model.predict(test_images)
    return np.argmax(predictions, axis=1)

def load_labels(label_csv):
    label_df = pd.read_csv(label_csv)
    return {row['class_index']: row['class'] for _, row in label_df.iterrows()}

def display(images, title, predicted_labels, label_dict, num_images=5): # Displays the modified images with predicted labels.
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        label = label_dict.get(predicted_labels[i], "Unknown")
        plt.title(f"{title}\nPred: {label}")
        plt.axis('off')
    plt.show()

def rotate_image(img, angle=20):
    return rotate(img, angle, reshape=False)



# Add more modification functions here



# Calculates what % of predicted labels on the modified images 
# are the same as the predictions on the original labels
def consistency_score(orig_preds, mod_preds):
    return sum(1 for o, m in zip(orig_preds, mod_preds) if o == m) / len(orig_preds) * 100

def test(model_path, test_dir, label_csv, img_size=(64, 64)):
    model = load_model(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    label_dict = load_labels(label_csv)
    test_images = []
    test_image_names = []
    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        img = preprocess(img_path, img_size)
        test_images.append(img)
        test_image_names.append(img_name)
    
    test_images = np.array(test_images)
    print("Evaluating original images:")
    original_preds = predict(model, test_images)
    display(test_images, "Original", original_preds, label_dict)
    results = {}
    modifications = {
        "Rotated": np.array([rotate_image(img) for img in test_images]),
       
       

       # Add the modifications here so they are displayed later



    }
    
    for mod_name, mod_images in modifications.items():
        print(f"\nEvaluating {mod_name} images:")
        mod_preds = predict(model, mod_images)
        display(mod_images, mod_name, mod_preds, label_dict)
        consistency = consistency_score(original_preds, mod_preds)
        results[mod_name] = consistency
        print(f"Consistency score (Original vs {mod_name}): {consistency:.2f}%")
    
    # Displays summary table of the 
    consistency_df = pd.DataFrame.from_dict(results, orient='index', columns=['Consistency score (%)'])
    print(consistency_df)
    return consistency_df

model_path = 'models/modelA.h5' # Change this to the model you want to test
results = test(
    model_path=model_path,
    test_dir='data/traffic_Data/TEST',
    label_csv='data/class_dict.csv',
    img_size=(64, 64)
)