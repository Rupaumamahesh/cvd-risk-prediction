"""
ONE-TIME script — converts the raw UCI file (processed.cleveland.data) into
a proper CSV with headers and a binary 'target' column.

Run from inside data/:
    python3 prepare_cleveland.py
"""

import pandas as pd

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

df = pd.read_csv("cleveland_raw/processed.cleveland.data", header=None, names=COLUMNS)

# UCI's 'num' column is 0 (no disease) to 4 (severity levels) — collapse
# to binary, matching your project's target definition (disease present or not)
df["target"] = (df["num"] > 0).astype(int)
df = df.drop(columns=["num"])

df.to_csv("cleveland.csv", index=False)
print(f"Saved data/cleveland.csv with {len(df)} rows, {df.shape[1]} columns")
print(df["target"].value_counts())
