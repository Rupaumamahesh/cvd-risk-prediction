"""
STEP 0 — Exploratory data visualization on the PRIMARY dataset (raw, before
cleaning), required for the submission rubric alongside preprocessing/training.

Input : data/cardio_train.csv
Output: outputs/viz_class_balance.png
        outputs/viz_age_distribution.png
        outputs/viz_bp_distributions.png
        outputs/viz_correlation_heatmap.png
        outputs/viz_boxplots_by_target.png

Run:
    python3 00_visualize_data.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_raw() -> pd.DataFrame:
    df = pd.read_csv("data/cardio_train.csv", sep=";")
    if df["age"].max() > 200:  # convert days -> years, same as Step 1
        df["age_years"] = (df["age"] / 365).round().astype(int)
    else:
        df["age_years"] = df["age"]
    return df

def plot_class_balance(df: pd.DataFrame):
    counts = df["cardio"].value_counts().sort_index()
    plt.figure(figsize=(5, 4))
    plt.bar(["No CVD (0)", "CVD (1)"], counts.values, color=["#4A8484", "#B85C5C"])
    for i, v in enumerate(counts.values):
        plt.text(i, v + 500, f"{v:,}", ha="center")
    plt.title("Class Balance — Primary Dataset")
    plt.ylabel("Number of patients")
    plt.tight_layout()
    plt.savefig("outputs/viz_class_balance.png", dpi=150)
    plt.close()
    print("Saved: outputs/viz_class_balance.png")

def plot_age_distribution(df: pd.DataFrame):
    plt.figure(figsize=(6, 4))
    plt.hist(df["age_years"], bins=30, color="#4A8484", edgecolor="white")
    plt.title("Age Distribution — Primary Dataset")
    plt.xlabel("Age (years)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("outputs/viz_age_distribution.png", dpi=150)
    plt.close()
    print("Saved: outputs/viz_age_distribution.png")

def plot_bp_distributions(df: pd.DataFrame):
    # Raw data has extreme outliers (e.g. BP of thousands) - clip just for
    # a readable plot; the actual cleaning happens in Step 1, not here.
    ap_hi = df["ap_hi"].clip(60, 240)
    ap_lo = df["ap_lo"].clip(40, 200)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(ap_hi, bins=30, color="#B85C5C", edgecolor="white")
    axes[0].set_title("Systolic BP (ap_hi)")
    axes[0].set_xlabel("mmHg")
    axes[1].hist(ap_lo, bins=30, color="#8C6BAE", edgecolor="white")
    axes[1].set_title("Diastolic BP (ap_lo)")
    axes[1].set_xlabel("mmHg")
    plt.tight_layout()
    plt.savefig("outputs/viz_bp_distributions.png", dpi=150)
    plt.close()
    print("Saved: outputs/viz_bp_distributions.png")

def plot_correlation_heatmap(df: pd.DataFrame):
    features = [
        "age_years", "gender", "height", "weight", "ap_hi", "ap_lo",
        "cholesterol", "gluc", "smoke", "alco", "active", "cardio",
    ]
    corr = df[features].corr()

    plt.figure(figsize=(8, 7))
    im = plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(features)), features, rotation=45, ha="right")
    plt.yticks(range(len(features)), features)
    for i in range(len(features)):
        for j in range(len(features)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.title("Feature Correlation Heatmap — Primary Dataset")
    plt.tight_layout()
    plt.savefig("outputs/viz_correlation_heatmap.png", dpi=150)
    plt.close()
    print("Saved: outputs/viz_correlation_heatmap.png")

def plot_boxplots_by_target(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    features = ["age_years", "cholesterol", "ap_hi"]
    titles = ["Age by CVD outcome", "Cholesterol by CVD outcome", "Systolic BP by CVD outcome"]

    for ax, feat, title in zip(axes, features, titles):
        data_0 = df[df["cardio"] == 0][feat].clip(upper=df[feat].quantile(0.99))
        data_1 = df[df["cardio"] == 1][feat].clip(upper=df[feat].quantile(0.99))
        ax.boxplot([data_0, data_1], tick_labels=["No CVD", "CVD"])
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig("outputs/viz_boxplots_by_target.png", dpi=150)
    plt.close()
    print("Saved: outputs/viz_boxplots_by_target.png")

def main():
    df = load_raw()
    print(f"Loaded {len(df)} raw records for visualization\n")

    plot_class_balance(df)
    plot_age_distribution(df)
    plot_bp_distributions(df)
    plot_correlation_heatmap(df)
    plot_boxplots_by_target(df)

    print("\nAll visualizations saved to outputs/")

if __name__ == "__main__":
    main()
