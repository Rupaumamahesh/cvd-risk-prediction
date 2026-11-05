# Additional machine-learning methods

Implemented LinearSVM, KNN and GaussianNB alongside the original Dummy, LogisticRegression, RandomForest and XGBoost. Main 70/15/15 split, 14 source predictors, seed 42 and stratified five-fold training CV are retained. No external datasets are used in this extension. No resampling, class weights, early stopping, calibration or ensemble-of-different-algorithms is added.

Numeric inputs use training-fitted median imputation and standardization. Categorical inputs use training-fitted most-frequent imputation and one-hot encoding with unknown categories ignored. Every candidate's preprocessing is inside its CV pipeline. This is an algorithm-and-preprocessing comparison: original models retain their original preprocessing, while new algorithms use categorical encoding.

Linear SVM uses squared hinge loss, L2 regularization, a linear boundary and maximum 20,000 iterations. C=[0.01,0.1,1] is tuned. Decision margins are valid ranking scores for ROC-AUC/average precision but are not probabilities; classification uses a zero margin and probability losses are omitted. A kernel SVM is omitted to control runtime on 47,968 training records.

KNN uses brute-force Euclidean distances, k=[15,51,101] and uniform/distance voting. It is sensitive to scales, categorical representation and correlated predictors; prediction retains and searches training records, so its saved artifact and inference cost can be larger.

GaussianNB tunes var_smoothing=[1e-9,1e-6,1e-3]. It is an inexpensive probabilistic reference, not a well-specified clinical generative model: correlated/derived features violate conditional independence, and one-hot columns are not Gaussian. Predicted probabilities should not be assumed calibrated.

Extended validation winner: XGBoost. Its test result is exploratory. Reused original winner test predictions; no new test inference. The primary model/results and previous report remain unchanged. Repeated use of validation and previously inspected test records does not yield independent confirmation. Training CV mean ± population SD reflects tuning folds, not nested-CV uncertainty or confidence intervals. No claim of statistical superiority, subject independence or clinical readiness is made.

Outputs: all_methods_validation.csv (seven models, discrimination, classification and probability metrics), new_methods_validation.csv, cv_*.csv, validation_predictions_*.csv, model_comparison.png, validation_roc_curves.png, extended_winner_test_results.csv, selection.json and verification.json. Fitted pipelines: models/additional_methods/.

Reproduce: venv/bin/python 07_compare_additional_methods.py (CVD_JOBS defaults to 2). The original training script remains unchanged.

Method references:
- https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
- https://scikit-learn.org/stable/modules/neighbors.html
- https://scikit-learn.org/stable/modules/naive_bayes.html
