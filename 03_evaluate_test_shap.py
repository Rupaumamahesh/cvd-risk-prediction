"""
STEP 3 — Final, one-time evaluation of the SELECTED model on the held-out
PRIMARY test set, plus SHAP explainability (global + one patient example).

Input : data/primary_test.csv, models/best_model.joblib
Output: outputs/test_results.csv, outputs/shap_summary.png, outputs/shap_waterfall_example.png

Run:
    python 03_evaluate_test_shap.py
"""

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

TARGET = "cardio"

def main():
    model = joblib.load("models/best_model.joblib")
    features = joblib.load("models/feature_list.joblib")
    test = pd.read_csv("data/primary_test.csv")

    X_test, y_test = test[features], test[TARGET]

    # --- Final honest metrics (looked at ONCE) ---
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    print("=== Final test-set metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    pd.DataFrame([metrics]).to_csv("outputs/test_results.csv", index=False)

    # --- SHAP: global feature importance ---
    print("\nComputing SHAP values (this can take a minute)...")
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)

    # Random Forest is a binary classifier, so SHAP returns explanations
    # shaped (samples, features, classes). We explicitly select class 1
    # ("has CVD") for both plots below, rather than let SHAP guess.
    shap_values_cvd = shap_values[:, :, 1]

    plt.figure()
    shap.summary_plot(shap_values_cvd, X_test, show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png", dpi=150)
    plt.close()
    print("Saved: outputs/shap_summary.png")

    # --- SHAP: one individual patient explanation ---
    plt.figure()
    shap.plots.waterfall(shap_values_cvd[0], show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_waterfall_example.png", dpi=150)
    plt.close()
    print("Saved: outputs/shap_waterfall_example.png")

if __name__ == "__main__":
    main()