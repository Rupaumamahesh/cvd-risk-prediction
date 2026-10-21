# Additional Review 2 validation

The main 70/15/15 experiment and selected model are unchanged.

## 80/10/10 check

Same cleaning and 14 predictors; stratified seed-42 random row split, 80% training and 10% each validation/test. Exact counts and class distributions: split_80_10_10_audit.json. Separate baseline and frozen-winner metrics: split_80_10_10_results.csv. No duplicate predictor profiles overlap. Pipelines are fitted afresh on the 80% training subset only.

This is an exploratory sensitivity check using previously inspected source records and parameters selected in the main experiment. It is not a new independent confirmation, and it is not used to choose the official split or model. Subject independence remains unverifiable.

## Repeated Cleveland CV

5 stratified folds × 10 repeats = 50 fits per model, seed 42, restricted to the original Cleveland training partition (212 records). The original 91-record holdout is excluded. The primary-selected algorithm/parameters are fixed; no external tuning occurs. Every fold fits its own imputer (and scaler for LR), with fitted medians checked against that fold's training data. Both Dummy and selected-model results are saved.

Selected-model ROC-AUC: 0.8655 ± 0.0506. SD is across 50 dependent fold scores, not a confidence interval or standard error. Individual repeats/folds and other metrics are in cleveland_repeated_cv_folds.csv. This supplements the existing single five-fold CV result; it does not replace or pool holdout metrics. Cleveland remains a separately retrained approach-replication experiment.
