# traffic_analysis_horizontal.py
"""Switch‑axis version of your traffic‑sign robustness notebook.

* Every grouped bar chart is now horizontal (`ax.barh`).
* Counts are on the **x‑axis**; models / class labels run down the **y‑axis**.
* Otherwise logic and file layout stay the same so you can drop it in place of
  the original script.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image  # noqa: F401 – kept for future image ops

# ---------------------------------------------------------------------------
# 1. Data helpers
# ---------------------------------------------------------------------------

def load_and_process_data(file_path: str):
    """Return a summary dict for a single model‑prediction CSV."""
    df = pd.read_csv(file_path)
    model_name = os.path.basename(file_path).split("_")[-1].split(".")[0]

    modification_columns = df.columns[3:]

    # mark every prediction as correct / wrong for each modification
    for col in modification_columns:
        df[f"{col} Correct"] = df[col] == df["True label"]

    results_per_modification = {}
    for col in modification_columns:
        results_per_modification[col] = {
            "originally_correct_still_correct": len(
                df[(df["True label"] == df["Original pred"]) & (df[col] == df["True label"])]
            ),
            "originally_wrong_still_wrong": len(
                df[(df["True label"] != df["Original pred"]) & (df[col] != df["True label"])]
            ),
            "originally_correct_became_wrong": len(
                df[(df["True label"] == df["Original pred"]) & (df[col] != df["True label"])]
            ),
            "originally_wrong_became_correct": len(
                df[(df["True label"] != df["Original pred"]) & (df[col] == df["True label"])]
            ),
        }

    return {
        "model": model_name,
        "results_per_modification": results_per_modification,
    }


# ---------------------------------------------------------------------------
# 2. Plot helpers – horizontal grouped bars
# ---------------------------------------------------------------------------

def plot_horizontal(results):
    """Grouped horizontal bars for each modification‑type sub‑figure."""

    modification_types = list(results[0]["results_per_modification"].keys())
    n_mod = len(modification_types)
    rows, cols = 2, 3  # unchanged grid layout
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()

    models = [r["model"] for r in results]
    y = np.arange(len(models))
    bar_height = 0.2

    for i, modi in enumerate(modification_types):
        if i >= len(axes):
            break
        ax = axes[i]

        # collect counts
        counts_cc = [r["results_per_modification"][modi]["originally_correct_still_correct"] for r in results]
        counts_wc = [r["results_per_modification"][modi]["originally_wrong_became_correct"] for r in results]
        counts_ww = [r["results_per_modification"][modi]["originally_wrong_still_wrong"] for r in results]
        counts_cw = [r["results_per_modification"][modi]["originally_correct_became_wrong"] for r in results]

        # plot – note offsets along *y* now
        ax.barh(y + bar_height * 1.5, counts_cc, height=bar_height, label="Orig. correct → still correct", alpha=0.7)
        ax.barh(y + bar_height / 2, counts_wc, height=bar_height, label="Orig. wrong → became correct", alpha=0.7)
        ax.barh(y - bar_height / 2, counts_ww, height=bar_height, label="Orig. wrong → still wrong", alpha=0.7)
        ax.barh(y - bar_height * 1.5, counts_cw, height=bar_height, label="Orig. correct → became wrong", alpha=0.7)

        ax.set_yticks(y)
        ax.set_yticklabels(models)
        ax.set_xlabel("Count")
        ax.set_title(modi)
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        if i == 0:
            ax.legend(fontsize="x-small")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3. Training‑data distribution – horizontal bar chart
# ---------------------------------------------------------------------------

def plot_training_distribution(data_dir: str, label_map: dict):
    """Plot training‑set class imbalance with horizontal bars."""

    label_counts = {}
    for label in os.listdir(data_dir):
        p = os.path.join(data_dir, label)
        if not os.path.isdir(p):
            continue
        n_img = len([f for f in os.listdir(p) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        label_name = label_map.get(label, f"Class {label}")
        label_counts[label_name.title()] = n_img

    # sort by frequency desc
    labels, counts = zip(*sorted(label_counts.items(), key=lambda x: x[1], reverse=True))

    avg = sum(counts) / len(counts)

    plt.figure(figsize=(12, 14))
    bars = plt.barh(labels, counts, color="#5B9BD5", edgecolor="black")
    plt.axvline(avg, color="red", linestyle="--", linewidth=1.5, label=f"Average = {avg:.1f}")

    plt.xlabel("Number of Training Images", fontsize=12)
    plt.ylabel("Class Label", fontsize=12)
    plt.title("Training Data Distribution", fontsize=14, weight="bold", pad=6)

    # annotate counts
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            plt.text(w + 1, bar.get_y() + bar.get_height() / 2, f"{int(w)}",
                     va="center", fontsize=10)

    plt.legend()
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.box(False)
    plt.show()


# ---------------------------------------------------------------------------
# 4. Main (example invocation)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None  # quiet SettingWithCopy warnings

    # your CSVs
    files = [
        "detailed_predictions_modelA.csv",
        "detailed_predictions_modelB.csv",
        "detailed_predictions_modelC.csv",
        "detailed_predictions_modelD.csv",
    ]

    # paths / label maps – adjust if needed
    CLASS_DICT_PATH = "data/class_dict.csv"  # unused downstream but kept
    DATA_DIR = "data/traffic_Data/DATA"
    LABEL_CSV_PATH = "data/traffic_Data/labels.csv"

    label_df = pd.read_csv(LABEL_CSV_PATH)
    label_map = dict(zip(label_df["ClassId"].astype(str), label_df["Name"]))

    # load model results
    results = [load_and_process_data(f) for f in files]

    # visualise robustness – horizontal bars
    plot_horizontal(results)

    # visualise class imbalance – horizontal bars
    plot_training_distribution(DATA_DIR, label_map)
