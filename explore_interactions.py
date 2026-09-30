import joblib
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

model = joblib.load("models/best_model.joblib")
feature_list = joblib.load("models/feature_list.joblib")
X_test = pd.read_csv("data/primary_test.csv")

X_sample = X_test[feature_list].sample(200, random_state=42)

explainer = shap.TreeExplainer(model)
shap_interaction_values = explainer.shap_interaction_values(X_sample)

print("Shape:", shap_interaction_values.shape)

interaction_pos = shap_interaction_values[:, :, :, 1]

shap.summary_plot(interaction_pos, X_sample, max_display=11, show=False)
plt.gcf().set_size_inches(20, 20)
plt.savefig("outputs/shap_interaction_summary.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved: outputs/shap_interaction_summary.png")