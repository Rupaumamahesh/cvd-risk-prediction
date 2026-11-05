# Explainable CVD classification — corrected Review 2 pipeline

Use the project virtual environment and run from this directory:

```bash
venv/bin/python 01_preprocess_primary.py
venv/bin/python 02_train_and_tune.py
venv/bin/python 03_evaluate_test_shap.py
venv/bin/python 04_generalization_test.py
venv/bin/python 05_review_artifacts.py
venv/bin/python verify_pipeline.py
venv/bin/python 06_additional_validation.py
```

Install packages from `requirements-lock.txt` to reproduce the executed environment. `requirements.txt` is the unpinned dependency list. Grid search defaults to four concurrent CPU jobs; set `CVD_JOBS` to adjust.

Datasets: [Kaggle CVD](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset), [UCI Cleveland](https://archive.ics.uci.edu/dataset/45/heart+disease), and [Framingham](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset). Raw inputs are `data/cardio_train.csv` (semicolon-separated), `data/cleveland.csv` (binary target `target`), and `data/framingham.csv` (`TenYearCHD`). No downloads occur during execution.

The corrected primary cohort contains 68,528 rows and 14 predictors (11 original + BMI, pulse pressure, MAP). Exact repeated original predictor profiles are excluded before splitting, including conflicting labels; ID is not used to identify duplicates. Age is retained at fractional-year precision. A stratified seed-42 split produces 47,968 training / 10,280 validation / 10,280 test records. There is no primary resampling. Missing values and LR scaling are fitted in pipelines inside training CV.

Dummy, scaled Logistic Regression, Random Forest and XGBoost are compared. Training-only five-fold grid search chooses each candidate's parameters; validation ROC-AUC chooses the algorithm. Full grids and per-fold scores are saved in `outputs/cv_*.csv`. Mean ± SD describes tuning CV, not nested CV. Only the selected pipeline is evaluated on primary test data.

The corrected winner is XGBoost (depth 3, 100 trees, learning rate 0.1): validation ROC-AUC 0.8025; training CV ROC-AUC 0.8005 ± 0.0051; test ROC-AUC 0.8028. The source test data was inspected in the previous pipeline, so this corrected rerun must not be presented as a fresh confirmatory test.

Cleveland and Framingham are separate within-dataset retraining experiments using the primary-selected algorithm and parameters, with 70/30 splits and training-fitted imputation. They do not validate transfer of the fixed fitted primary model. Their outcomes and predictor sets differ. External holdout ROC-AUC: Cleveland 0.9052, Framingham 0.6816. Framingham recall is only 0.0933 at threshold 0.5; high accuracy must not hide this limitation. RF/LR use balanced training class weights externally; the selected XGBoost uses no weighting. External dummy baselines are included.

Review materials (no PPT/report changes):

- `outputs/review2_setup.md`: experimental setup and interpretation notes.
- `outputs/preprocessing_audit.json`, `external_audit.json`: counts, distributions and leakage controls.
- `outputs/environment.json`, `hardware.json`, `fitted_hyperparameters.json`: measured setup and exact estimator settings.
- `outputs/validation_results.csv`, `test_results.csv`, `generalization_results.csv`: actual results.
- `outputs/roc_curve.png`, `confusion_matrix.png`, `shap_summary.png`, `class_distribution.png`: four meaningful charts.
- `outputs/model_comparison.png`, `cross_dataset_comparison.png`: comparison charts.
- `outputs/shap_waterfall_example.png`, `shap_importance.csv`, `shap_metadata.json`: patient and global explanations.
- `outputs/architecture.drawio`: editable architecture generated inside the online draw.io editor; `architecture_drawio_preview.png` shows the editor.

SHAP uses a seed-42 sample of 500 test rows. XGBoost uses native exact TreeSHAP in log-odds units, with training tree-cover statistics; contribution sums are checked against raw prediction margins. Other supported winners use the training-background SHAP path. The plot is an explicit beeswarm, not a multiclass importance bar plot.

Subject-level separation cannot be verified from these files because no reliable repeated-patient identifier is available. Excluding identical measured profiles prevents exact profile overlap but does not establish subject independence. No temporal or sequence validation claim is made. No feature selection or test-driven threshold tuning is performed.

Additional requested checks: `06_additional_validation.py` performs an exploratory 80/10/10 split sensitivity evaluation with the frozen selected configuration, plus Cleveland training-only 5-fold CV repeated 10 times. It preserves the main experiment. Counts, baseline metrics, all 50 fold scores per model and interpretation notes are saved in `outputs/additional_validation/`. These reused-data split results are not independent confirmation; repeated CV SD is not a confidence interval.

## Additional algorithms

Run `venv/bin/python 07_compare_additional_methods.py` to tune **linear SVM, K-Nearest Neighbours and Gaussian Naive Bayes** using the existing primary training split and stratified five-fold CV. The script compares their validation results with the original four candidates and saves separate models, grids, fold scores, charts and a frozen extended-winner evaluation under `models/additional_methods/` and `outputs/additional_methods/`. It does not overwrite the original model/results, SHAP figures or report. Run `venv/bin/python verify_additional_methods.py` after completion to verify the artifacts.

New pipelines standardize numeric inputs and one-hot encode categorical inputs, with fitted imputation and encoding inside CV. Linear SVM scores are margins, not probabilities. GaussianNB is an approximate baseline because conditional independence/Gaussian assumptions do not hold exactly. KNN is sensitive to correlated predictors and distance representation. Expanded comparison is exploratory development on previously inspected source records and reused validation data, not fresh independent confirmation. No ensemble or external-dataset rerun is included in this extension.

## Heterogeneous ensembles

Run `venv/bin/python 08_train_ensembles.py` to compare **equal-weight soft voting** and **stacking** of XGBoost, Random Forest and KNN against individual XGBoost. Stacking uses a regularized Logistic Regression combiner fitted on training out-of-fold probabilities. Five outer training folds evaluate the full inner-five-fold procedure and select the combiner's C. Base hyperparameters remain frozen from earlier searches; outer CV results are exploratory tuning estimates, not fully nested base-parameter selection. After training, run `venv/bin/python verify_ensembles.py`.

Artifacts are separate: `outputs/ensembles/` and `models/ensembles/`. These include complete CV scores, inner-fold preprocessing audits, training OOF diversity diagnostics, validation metrics, ROC/precision–recall/calibration charts and a validation-selected model's exploratory test result. Existing individual models, reports and SHAP figures remain unchanged. The original XGBoost SHAP plots do not explain the combined ensemble. Neither validation nor test data fits any ensemble layer or preprocessing. Reused validation/source test data prevent independent confirmatory claims.

## Refresh numeric chart labels without retraining

Run `venv/bin/python 09_refresh_plot_labels.py`. Plotting functions in scripts 00, 03, 05, 07 and 08 now display counts, bar values, CV mean ± SD and four-decimal AUC/AP legends where appropriate. Boxplots include quartile tables. The SHAP beeswarm labels each feature with its mean absolute contribution instead of cluttering every dot. Calibration coordinates are shown in a separate numbered-bin chart/table (`outputs/ensembles/calibration_numeric.png`). The interaction chart uses native current-XGBoost TreeSHAP interactions on 200 fixed test records and labels every matrix cell; it replaces an obsolete Random Forest-specific plotting script.

The refresh fits no models. It checks that all existing metric, prediction, CV, selection and fitted-model artifacts remain byte-identical; numeric display exports are allowed to update. The architecture editor screenshot is not a quantitative chart and remains unchanged. Existing DOCX/PDF reports are not regenerated by this command. Verification: `outputs/plot_label_verification.json`.
