"""
STEP 1 — Clean and split the PRIMARY dataset (Kaggle Cardiovascular Disease Dataset).

Input : data/cardio_train.csv  (download from Kaggle, ';'-separated)
Output: data/primary_train.csv, data/primary_val.csv, data/primary_test.csv

Run:
    python 01_preprocess_primary.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/cardio_train.csv"
AGE_IN_DAYS = True  # Kaggle dataset stores age in days, not years

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")

    # Convert age from days to years (Kaggle quirk) for interpretability
    if AGE_IN_DAYS and df["age"].max() > 200:
        df["age"] = (df["age"] / 365).round().astype(int)

    before = len(df)

    # --- Remove physiologically implausible values ---
    df = df[(df["ap_hi"] > 60) & (df["ap_hi"] < 240)]      # systolic BP
    df = df[(df["ap_lo"] > 40) & (df["ap_lo"] < 200)]      # diastolic BP
    df = df[df["ap_hi"] >= df["ap_lo"]]                    # systolic must exceed diastolic
    df = df[(df["height"] > 100) & (df["height"] < 220)]   # cm
    df = df[(df["weight"] > 30) & (df["weight"] < 200)]    # kg

    # --- Drop duplicates ---
    df = df.drop_duplicates()

    after = len(df)
    print(f"Cleaning removed {before - after} rows ({before} -> {after})")

    return df

def split_and_save(df: pd.DataFrame):
    target = "cardio"

    # Split off test set first (15%), then split remainder into train/val
    train_val, test = train_test_split(
        df, test_size=0.15, stratify=df[target], random_state=42
    )
    train, val = train_test_split(
        train_val, test_size=0.176, stratify=train_val[target], random_state=42
    )  # 0.176 of the remaining 85% ~= 15% of the original total

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print("Class balance (train):")
    print(train[target].value_counts(normalize=True))

    train.to_csv("data/primary_train.csv", index=False)
    val.to_csv("data/primary_val.csv", index=False)
    test.to_csv("data/primary_test.csv", index=False)
    print("Saved: data/primary_train.csv, primary_val.csv, primary_test.csv")

if __name__ == "__main__":
    df = load_and_clean(RAW_PATH)
    split_and_save(df)
