"""
STEP 4 — Generalization testing on Cleveland and Framingham.

IMPORTANT (per project discussion): these datasets do NOT share the primary
dataset's 11 columns, so the primary model literally cannot be reused as-is.
Instead we reuse the SAME algorithm + SAME tuned hyperparameters (winner from
Step 2), but fit a fresh instance on each dataset's own feature set. This
tests whether the *modeling approach* generalizes, not whether one fixed
model object can be reused untouched.

Input : data/cleveland.csv, data/framingham.csv, outputs/validation_results.csv
Output: outputs/generalization_results.csv

Run:
    python 04_generalization_test.py
"""

import ast
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.impute import SimpleImputer

MODEL_LOOKUP = {
    "LogisticRegression": lambda p: LogisticRegression(
        max_iter=2000, class_weight="balanced", **p
    ),
    "RandomForest": lambda p: RandomForestClassifier(
        random_state=42, class_weight="balanced", **p
    ),
    "XGBoost": lambda p: XGBClassifier(
        random_state=42, use_label_encoder=False, eval_metric="logloss", **p
    ),
}

def get_winning_config():
    """Reads which model + hyperparameters won on the primary dataset."""
    results = pd.read_csv("outputs/validation_results.csv")
    winner_row = results.sort_values("roc_auc", ascending=False).iloc[0]
    name = winner_row["model"]
    params = ast.literal_eval(winner_row["best_params"])
    print(f"Reusing winning config from primary dataset: {name} {params}")
    return name, params

def evaluate(model, X, y) -> dict:
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
        "roc_auc": roc_auc_score(y, proba),
    }

def run_on_dataset(name: str, path: str, target_col: str, model_name: str, params: dict) -> dict:
    df = pd.read_csv(path)
    df = df.replace("?", pd.NA)  # Cleveland uses '?' for missing values

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce")

    # Impute remaining missing values (median) rather than dropping rows,
    # since Cleveland is small (303 rows) and every row matters.
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = MODEL_LOOKUP[model_name](params)
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    metrics["dataset"] = name
    metrics["n_records"] = len(df)
    metrics["n_features"] = X.shape[1]
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    return metrics

def main():
    model_name, params = get_winning_config()

    all_results = []
    all_results.append(
        run_on_dataset("Cleveland", "data/cleveland.csv", "target", model_name, params)
    )
    all_results.append(
        run_on_dataset("Framingham", "data/framingham.csv", "TenYearCHD", model_name, params)
    )

    out = pd.DataFrame(all_results)
    out.to_csv("outputs/generalization_results.csv", index=False)
    print("\nSaved: outputs/generalization_results.csv")
    print("\nReminder: metrics are reported PER DATASET, never pooled together.")

if __name__ == "__main__":
    main()