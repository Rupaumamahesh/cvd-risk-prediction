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
    fig, ax = plt.subplots(figsize=(10, 5))
    counts, edges, bars = ax.hist(df['age_years'], bins=30, color='#4A8484', edgecolor='white')
    ax.bar_label(bars, labels=[f'{int(v):,}' if v else '' for v in counts], rotation=90, padding=3, fontsize=8)
    ax.set(title='Age Distribution — Primary Dataset', xlabel='Age (years)', ylabel='Number of records', ylim=(0,max(counts)*1.25))
    fig.tight_layout(); fig.savefig('outputs/viz_age_distribution.png', dpi=250); plt.close(fig)
    pd.DataFrame({'bin_lower_years':edges[:-1],'bin_upper_years':edges[1:],'count':counts.astype(int)}).to_csv('outputs/viz_age_bin_counts.csv',index=False)
    print('Saved: outputs/viz_age_distribution.png')

def plot_bp_distributions(df: pd.DataFrame):
    # Raw data has extreme outliers (e.g. BP of thousands) - clip just for
    # a readable plot; the actual cleaning happens in Step 1, not here.
    ap_hi = df["ap_hi"].clip(60, 240)
    ap_lo = df["ap_lo"].clip(40, 200)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bin_tables = []
    for ax, values, title, color in [
        (axes[0], ap_hi, "Systolic BP (ap_hi)", "#B85C5C"),
        (axes[1], ap_lo, "Diastolic BP (ap_lo)", "#8C6BAE"),
    ]:
        counts, edges, bars = ax.hist(values, bins=30, color=color, edgecolor="white")
        # Label every nonempty bin with its exact record count, including small bars.
        ax.bar_label(
            bars, labels=[f"{int(n):,}" if n else "" for n in counts],
            padding=3, rotation=90, fontsize=8,
        )
        ax.set_title(title)
        ax.set_xlabel("Blood pressure (mmHg)")
        ax.set_ylabel("Number of records")
        ax.set_ylim(0, max(counts) * 1.25)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        assert int(counts.sum()) == len(values.dropna())
        bin_tables.append(pd.DataFrame({
            "measurement": title, "bin_lower_mmHg": edges[:-1],
            "bin_upper_mmHg": edges[1:], "count": counts.astype(int),
        }))
    fig.suptitle("Raw blood pressure distributions — exact counts above nonempty bins", fontsize=12)
    fig.text(0.5, 0.02,
             "Clipped for display only: systolic 60–240 mmHg; diastolic 40–200 mmHg. "
             "Out-of-range values contribute to boundary bins.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig("outputs/viz_bp_distributions.png", dpi=300)
    fig.savefig("outputs/viz_bp_distributions.svg")
    plt.close(fig)
    pd.concat(bin_tables, ignore_index=True).to_csv("outputs/viz_bp_bin_counts.csv", index=False)
    print("Saved: outputs/viz_bp_distributions.png (and SVG / exact bin counts)")

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
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    features = ["age_years", "cholesterol", "ap_hi"]
    titles = ["Age by CVD outcome", "Cholesterol by CVD outcome", "Systolic BP by CVD outcome"]

    for ax, feat, title in zip(axes, features, titles):
        data_0 = df[df["cardio"] == 0][feat].clip(upper=df[feat].quantile(0.99))
        data_1 = df[df["cardio"] == 1][feat].clip(upper=df[feat].quantile(0.99))
        ax.boxplot([data_0, data_1], tick_labels=["No CVD", "CVD"])
        ax.set_title(title)
        quartiles = [np.quantile(v, [.25,.5,.75]) for v in [data_0,data_1]]
        table = ax.table(cellText=[[f'{quartiles[0][i]:.2f}',f'{quartiles[1][i]:.2f}'] for i in range(3)],
                         rowLabels=['Q1','Median','Q3'],colLabels=['No CVD','CVD'],
                         cellLoc='center',bbox=[.10,-.54,.90,.38])
        table.auto_set_font_size(False);table.set_fontsize(9)

    fig.subplots_adjust(bottom=.36,top=.9,wspace=.35)
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
