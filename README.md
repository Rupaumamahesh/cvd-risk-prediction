# Explainable CVD Risk Prediction — Execution Guide

Matches the pipeline in your Project Review-1 deck (Methodology, Generalization
Testing, and Training Parameters slides).

## 0. Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Get the three datasets

| Dataset | Where to get it | Save as |
|---|---|---|
| Primary — Kaggle CVD | https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset | `data/cardio_train.csv` |
| Cleveland — UCI Heart Disease | https://archive.ics.uci.edu/dataset/45/heart+disease | `data/cleveland.csv` |
| Framingham — 10-Year CHD | https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset | `data/framingham.csv` |

For Cleveland, rename the target column to `target` (1 = disease present).
For Framingham, keep the target column named `TenYearCHD` (already default).

## 2. Run the pipeline, in order

```bash
python 01_preprocess_primary.py     # clean + split primary dataset
python 02_train_and_tune.py         # train & tune all 3 models, pick winner by val ROC-AUC
python 03_evaluate_test_shap.py     # final test metrics + SHAP plots for the winner
python 04_generalization_test.py    # evaluate the winning approach on Cleveland & Framingham
```

Each script reads the output of the one before it — run them in this order the
first time. `outputs/` will contain everything you need for your slides:

| File | Goes on slide |
|---|---|
| `outputs/validation_results.csv` | Methodology — "Compare (val)" / "Select best (ROC-AUC)" |
| `outputs/test_results.csv` | Methodology — "Test + SHAP" |
| `outputs/shap_summary.png` | Expected Project Output — SHAP visual explanation |
| `outputs/shap_waterfall_example.png` | Expected Project Output — per-patient explanation |
| `outputs/generalization_results.csv` | Generalization Testing slide |

## 3. What each step actually does (ties back to your slides)

1. **Preprocess** — removes implausible BP/height/weight values, drops
   duplicates, does the stratified train/val/test split (Validation row on
   your Methodology slide).
2. **Train & tune** — runs grid search with 5-fold cross-validation for each
   of the three models, using the exact search ranges from your Training
   Parameters slide, then picks the winner by **validation ROC-AUC only**
   (not accuracy — see the "why ROC-AUC" discussion from earlier).
3. **Test + SHAP** — the ONE honest look at the held-out test set, plus
   global (summary plot) and per-patient (waterfall plot) SHAP explanations.
4. **Generalization** — reuses the *same algorithm and hyperparameters* that
   won on the primary dataset, but fits a fresh instance on each of
   Cleveland's and Framingham's own feature sets (they don't share your
   primary dataset's 11 columns, so the literal same model object can't be
   reused — this is the corrected framing from your slide 17 discussion).
   Results are reported separately per dataset, never pooled.

## 4. Before your review — sanity checks

- [ ] Re-run Step 2 and note down the actual winning model + hyperparameters
      (replace the illustrative ranges on your Training Parameters slide with
      real ones)
- [ ] Check `validation_results.csv` — confirm ROC-AUC, not accuracy, decided
      the winner
- [ ] Open `shap_summary.png` — identify the top 3–5 features, have them
      ready to name out loud
- [ ] Check `generalization_results.csv` — if performance drops sharply on
      Cleveland/Framingham, that's expected and worth addressing directly
      rather than hiding (small-sample Cleveland and imbalanced Framingham
      are both harder settings by design)
