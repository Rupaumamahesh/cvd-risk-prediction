# Heterogeneous ensembles

Compared equal-weight SoftVoting and Stacking (XGBoost + RandomForest + KNN base pipelines, LogisticRegression combiner) against individual XGBoost on the original 70/15/15 split. Seed 42; 14 predictors. No class weighting, resampling, external dataset retraining, threshold selection or post-hoc calibration is added.

Base settings: XGBoost depth 3, 100 trees, learning rate 0.1; RF depth 10, 300 trees; KNN k=101 with uniform Euclidean voting. Original tree pipelines retain median imputation; KNN retains numeric median imputation/scaling and categorical most-frequent imputation/one-hot encoding. Every fitting subset has its own preprocessing.

Five outer training folds evaluate the complete stacking procedure. Within each outer training subset, five inner folds generate three out-of-fold probabilities per record. LogisticRegression combiners C=[0.01, 0.1, 1.0] fit these training OOF predictions and are evaluated on the outer held-out records, whose data never fit either layer or preprocessing. SoftVoting averages the same outer refitted members. Combiner C selected by mean outer-fold ROC-AUC: 1.0. These scores are tuning estimates (C selected on them); base settings were also previously tuned using the full primary training partition, so this is not fully nested hyperparameter selection. Fold SD is not a confidence interval.

Final fit generates new five-fold OOF probabilities from all 47,968 training records to fit the selected combiner, and refits each member on all training records. No original inputs are passed directly to the combiner. LogisticRegression has intercept and L2 regularization; coefficients need not sum to one or be positive. They are not feature-level clinical effects.

Validation winner: XGBoost. Reused saved original test predictions; no new test inference. Original XGBoost, other individual models, reports and prior results remain unchanged. Validation and test records were previously examined; these results are exploratory and are not independent confirmation. Subject independence cannot be verified from available IDs.

Validation ROC-AUC, recall, specificity, precision, F1, average precision, NPV, log loss and Brier score are saved. Calibration curves are diagnostics, not proof of calibration. Base OOF correlations and class disagreement quantify similarity; algorithmic diversity alone does not guarantee complementary errors. Ensemble improvements are observed differences, not claims of statistical superiority. The primary XGBoost SHAP plots explain only XGBoost, not the new ensemble.

Scripts: 08_train_ensembles.py; ensemble_models.py; verify_ensembles.py. Reproduce with venv/bin/python 08_train_ensembles.py (CVD_JOBS defaults to 2). Fitted inference artifacts are in models/ensembles/. FittedProbabilityEnsemble is an inference-only container; do not use it for prefit stacking CV.

References:
- Wolpert (1992), Stacked generalization, Neural Networks 5(2), 241–259: https://doi.org/10.1016/S0893-6080(05)80023-1
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
