import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

class_dict = pd.read_csv("data/class_dict.csv", index_col="class_index")
data_dir = "data/traffic_Data/DATA"
label_csv_path = "data/traffic_Data/labels.csv"
label_df = pd.read_csv(label_csv_path)
label_map = dict(zip(label_df["ClassId"].astype(str), label_df["Name"]))

def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    model_name = os.path.basename(file_path).split('_')[-1].split('.')[0]
    modification_columns = df.columns[3:]
    for col in modification_columns:
        df[f"{col} Correct"] = df[col] == df["True label"]

    results_per_modification = {}
    for col in modification_columns:
        results_per_modification[col] = {
            "originally_correct_still_correct": len(df[(df["True label"] == df["Original pred"]) & (df[col] == df["True label"])]),
            "originally_wrong_still_wrong": len(df[(df["True label"] != df["Original pred"]) & (df[col] != df["True label"])]),
            "originally_correct_became_wrong": len(df[(df["True label"] == df["Original pred"]) & (df[col] != df["True label"])]),
            "originally_wrong_became_correct": len(df[(df["True label"] != df["Original pred"]) & (df[col] == df["True label"])]),
        }
    return {
        "model": model_name,
        "results_per_modification": results_per_modification
    }

def plot(results):
    modification_types = list(results[0]["results_per_modification"].keys())
    num_modifications = len(modification_types)
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, modification in enumerate(modification_types):
        if i >= len(axes):
            break
        ax = axes[i]
        models = [r["model"] for r in results]
        originally_correct_still_correct = [r["results_per_modification"][modification]["originally_correct_still_correct"] for r in results]
        originally_wrong_still_wrong = [r["results_per_modification"][modification]["originally_wrong_still_wrong"] for r in results]
        originally_correct_became_wrong = [r["results_per_modification"][modification]["originally_correct_became_wrong"] for r in results]
        originally_wrong_became_correct = [r["results_per_modification"][modification]["originally_wrong_became_correct"] for r in results]
        bar_width = 0.2
        x = np.arange(len(models))
        ax.bar(x - bar_width * 1.5, originally_correct_still_correct, width=bar_width, label="Originally correct, still correct", color="blue", alpha=0.7)
        ax.bar(x - bar_width/2, originally_wrong_became_correct, width=bar_width, label="Originally wrong, became correct", color="green", alpha=0.7)
        ax.bar(x + bar_width/2, originally_wrong_still_wrong, width=bar_width, label="Originally wrong, still wrong", color="red", alpha=0.7)
        ax.bar(x + bar_width * 1.5, originally_correct_became_wrong, width=bar_width, label="Originally correct, became wrong", color="orange", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20)
        ax.set_ylabel("Count")
        ax.set_title(f"{modification}")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)


    plt.tight_layout()
    plt.show()

files = ["detailed_predictions_modelA.csv", "detailed_predictions_modelB.csv", "detailed_predictions_modelC.csv", "detailed_predictions_modelD.csv"]
results = [load_and_process_data(file) for file in files]
plot(results)

label_counts = {}
for label in os.listdir(data_dir):
    label_path = os.path.join(data_dir, label)
    if os.path.isdir(label_path):
        image_count = len([
            f for f in os.listdir(label_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        label_name = label_map.get(label, f"Class {label}")
        label_counts[label_name] = image_count

plt.rcParams["font.family"] = "Times New Roman"

label_counts_cap = {label.title(): count for label, count in label_counts.items()}
sorted_items = sorted(label_counts_cap.items(), key=lambda x: x[1], reverse=True)
sorted_labels = [item[0] for item in sorted_items]
sorted_counts = [item[1] for item in sorted_items]

average_count = sum(sorted_counts) / len(sorted_counts)

plt.figure(figsize=(20, 10))
bars = plt.bar(sorted_labels, sorted_counts, color="#5B9BD5", edgecolor="black", width=0.6)
plt.axhline(average_count, color='red', linestyle='--', linewidth=1.5, label=f'Average = {average_count:.1f}')
plt.xticks(rotation=75, ha='right', fontsize=12)
plt.ylabel("Number of Training Images", fontsize=14)
plt.xlabel("Class Label", fontsize=14)
plt.title("Training Data Distribution", fontsize=16, weight='bold', pad=3)

for bar in bars:
    yval = bar.get_height()
    if yval > 0:
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, str(yval),
                 ha='center', va='bottom', fontsize=12)

plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.box(False)
plt.show()


