# Review 2 experimental setup

This is supervised binary classification of recorded CVD status (0/1) in the primary dataset, not a validated forecast of future events. The Cleveland outcome is disease presence; Framingham predicts ten-year CHD. Their scores are separate experiments with different endpoints and features.

Primary data: Kaggle cardiovascular disease dataset, 70,000 raw rows. Final count: 68528. Details of exclusions and class counts: preprocessing_audit.json. No resampling or class weighting is used on the primary data. Class distribution before and after balancing is therefore unchanged.

11 raw predictors: age, gender, height, weight, systolic BP, diastolic BP, cholesterol, glucose category, smoking, alcohol, activity. ID is excluded. Age is converted from days to years using 365.25 without rounding. Three deterministic derived predictors give 14 total: BMI = kg/(height in metres)^2; pulse pressure = systolic minus diastolic BP; MAP = (systolic + 2×diastolic)/3. No feature selection is performed.

Random row-wise, outcome-stratified split: 70% TRAINING, 15% validation, 15% test (integer rounding applies). Seed/random_state = 42 for splits, shuffled stratified five-fold CV and stochastic estimators. No sequence, time window, device split or temporal split applies. ID is a record identifier, not a verified repeated-subject identifier: subject independence cannot be guaranteed. Exact repeated original predictor profiles are removed in full, including conflicting labels; cross-split profile overlap is asserted zero. This conservative policy can remove distinct people with identical measured profiles.

All median imputation and LR standardization are learned inside the training pipeline and each CV fold. Deterministic row cleaning and feature formulas do not estimate population statistics. Test and validation never fit preprocessing. No outcome-derived input is used.

Baseline: DummyClassifier(strategy=prior) and scaled Logistic Regression. Grid-search candidates: LR C=[0.01,0.1,1,10]; RF n_estimators=[100,200,300], max_depth=[5,10,15]; XGBoost learning_rate=[0.01,0.05,0.1], n_estimators=[100,200,300], max_depth=[3,5,7]. LR C is inverse regularization strength, not a random seed; selected through training CV. LR uses L2 regularization and lbfgs, max_iter=2000. RF uses bootstrap aggregation and Gini impurity. XGBoost uses binary logistic objective, log-loss evaluation, histogram trees and gradient boosting. LR optimizes regularized logistic loss. RF has no learning-rate parameter; XGBoost's learning rate is tuned. No deep learning or early stopping is used; tree depth, tree count and regularization constrain complexity.

Winner: XGBoost; selected parameters: {"model__learning_rate": 0.1, "model__max_depth": 3, "model__n_estimators": 100}. Validation ROC-AUC: 0.8025. Training CV ROC-AUC: 0.8005 ± 0.0051 (population SD across five folds). These folds also selected hyperparameters: they are tuning estimates, not nested-CV unbiased estimates. Final test ROC-AUC: 0.8028. Full metrics and baseline improvements are in validation_results.csv. Classification threshold remains 0.5.

The test set was examined in the previous pipeline. This is a corrected rerun on the same source data, not a new never-seen confirmatory evaluation. No test-driven hyperparameter changes were made in this corrected pipeline. A new independent cohort is required for a fresh confirmatory claim.

External experiments use the selected algorithm and hyperparameters, retrained separately on each dataset's own predictors, with stratified 70/30 train/test and seed 42. Median imputation is fitted after splitting and inside external training CV. RF/LR use training-only balanced class weights on external datasets; XGBoost uses no weighting. No external hyperparameter search is performed. These are approach-replication experiments, not external validation of a fixed fitted primary model. Different endpoints, predictors and small Cleveland size limit direct comparison.

SHAP uses a reproducible 500-record test sample. XGBoost uses native exact path-dependent TreeSHAP with training tree-cover statistics and a verified additive reconstruction of raw margins; RF/LR use 100 training background records. The global plot is an explicit beeswarm. Units depend on the selected estimator; see shap_metadata.json. Explanations describe model associations, not causal or clinical effects.

Hardware/software measured from this execution machine: Intel(R) Core(TM) i5-7360U CPU @ 2.30GHz, 4 logical CPUs, 16.0 GiB RAM, macOS-13.7.7-x86_64-i386-64bit-Mach-O. CPU training, four concurrent grid-search jobs. Exact package/Python versions: environment.json and requirements-lock.txt.

Charts: roc_curve.png, confusion_matrix.png, shap_summary.png, class_distribution.png (four meaningful charts); model_comparison.png (baseline comparison); cross_dataset_comparison.png (separate dataset comparison). Architecture source: architecture.drawio. PPT and report have not been updated.

Additional checks: see [80/10/10 sensitivity and repeated Cleveland CV](additional_validation/README.md). These supplement the main experiment without changing its official split or winner.
