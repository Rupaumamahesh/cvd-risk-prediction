"""
STEP 2 — Train + hyperparameter-tune all 3 models on the PRIMARY dataset,
then select the best one by VALIDATION ROC-AUC (not accuracy).

Input : data/primary_train.csv, data/primary_val.csv
Output: models/best_model.joblib, outputs/validation_results.csv

Run:
    python 02_train_and_tune.py
"""

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

TARGET = "cardio"
FEATURES = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active",
]

# --- Hyperparameter grids (matches the deck's "Training Parameters" slide) ---
PARAM_GRIDS = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=2000),
        "params": {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"]},
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15],
        },
    },
    "XGBoost": {
        "model": XGBClassifier(
            random_state=42, use_label_encoder=False, eval_metric="logloss"
        ),
        "params": {
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
        },
    },
}

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

def main():
    train = pd.read_csv("data/primary_train.csv")
    val = pd.read_csv("data/primary_val.csv")

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    fitted_models = {}

    for name, cfg in PARAM_GRIDS.items():
        print(f"\nTuning {name} ...")
        search = GridSearchCV(
            cfg["model"], cfg["params"], scoring="roc_auc", cv=cv, n_jobs=-1
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        fitted_models[name] = best

        metrics = evaluate(best, X_val, y_val)
        metrics["model"] = name
        metrics["best_params"] = search.best_params_
        results.append(metrics)
        print(f"  Best params: {search.best_params_}")
        print(f"  Validation ROC-AUC: {metrics['roc_auc']:.4f}")

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    results_df.to_csv("outputs/validation_results.csv", index=False)
    print("\n=== Validation comparison (sorted by ROC-AUC) ===")
    print(results_df[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]])

    # --- Selection rule: highest validation ROC-AUC wins ---
    winner_name = results_df.iloc[0]["model"]
    winner_model = fitted_models[winner_name]
    print(f"\nSelected model: {winner_name}")

    joblib.dump(winner_model, "models/best_model.joblib")
    joblib.dump(FEATURES, "models/feature_list.joblib")
    print("Saved: models/best_model.joblib")

if __name__ == "__main__":
    main()
