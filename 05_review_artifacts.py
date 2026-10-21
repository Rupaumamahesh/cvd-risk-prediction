"""Create review charts and factual experimental-setup notes (no slides/report)."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/cvd-matplotlib')
import json, platform, sys, subprocess
import joblib
from sklearn.metrics import RocCurveDisplay
from pipeline_common import FEATURES
from importlib.metadata import version
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    val=pd.read_csv('outputs/validation_results.csv')
    test=pd.read_csv('outputs/test_results.csv').iloc[0]
    ext=pd.read_csv('outputs/generalization_results.csv')
    audit=json.loads(Path('outputs/preprocessing_audit.json').read_text())
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(val.model,val.roc_auc,label='Validation ROC-AUC',alpha=.75)
    ax.errorbar(val.model,val.cv_roc_auc_mean,yerr=val.cv_roc_auc_std,fmt='ko',capsize=5,label='Training CV mean ± SD')
    ax.set(ylim=(.4,.9),ylabel='ROC-AUC',title='Model comparison: validation selection and training CV')
    ax.legend(); fig.tight_layout(); fig.savefig('outputs/model_comparison.png',dpi=180); plt.close(fig)
    external=ext.loc[ext.model.ne('Dummy')]
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    labels=['Primary',*external.dataset]
    axes[0].bar(labels,[test.roc_auc,*external.roc_auc]); axes[0].axhline(.5,color='black',linestyle='--'); axes[0].set(ylabel='Held-out ROC-AUC',ylim=(0,1))
    axes[1].bar(labels,[test.recall,*external.recall]); axes[1].set(ylabel='Recall at threshold 0.5',ylim=(0,1))
    fig.suptitle('Separate dataset results: external models retrained; endpoints differ')
    fig.tight_layout(); fig.savefig('outputs/cross_dataset_comparison.png',dpi=180); plt.close(fig)
    validation=pd.read_csv('data/primary_val.csv')
    fig,ax=plt.subplots(figsize=(8,6))
    for name in val.model:
        model=joblib.load(f'models/{name}.joblib')
        RocCurveDisplay.from_estimator(model,validation[FEATURES],validation.cardio,ax=ax,name=name)
    ax.plot([0,1],[0,1],'k--'); ax.set_title('Validation ROC curves: model selection only')
    fig.tight_layout(); fig.savefig('outputs/validation_roc_curves.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6))
    for dataset in ['Cleveland','Framingham']:
        pred=pd.read_csv(f'outputs/{dataset.lower()}_predictions.csv')
        RocCurveDisplay.from_predictions(pred.actual,pred.probability,ax=ax,name=dataset)
    ax.plot([0,1],[0,1],'k--'); ax.set_title('External holdouts: separately retrained models; different endpoints')
    fig.tight_layout(); fig.savefig('outputs/external_roc_curves.png',dpi=180); plt.close(fig)
    counts=pd.DataFrame({s:d['class_counts'] for s,d in audit['splits'].items()}).T
    counts.columns=['Class '+str(c) for c in counts.columns]
    ax=counts.plot.bar(stacked=True,figsize=(7,5)); ax.set(ylabel='Records',title='Natural class distribution; no resampling'); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig('outputs/class_distribution.png',dpi=180); plt.close()
    def hardware(key):
        try:
            return subprocess.check_output(['/usr/sbin/sysctl','-n',key],text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError):
            return json.loads(Path('outputs/hardware.json').read_text())[key]
    setup={'os':platform.platform(),'architecture':platform.machine(),'cpu':hardware('machdep.cpu.brand_string'),'logical_cpus':hardware('hw.ncpu'),'ram_gib':round(int(hardware('hw.memsize'))/2**30,2),'python':sys.version,'packages':{p:version(p) for p in ['pandas','numpy','scikit-learn','xgboost','shap','matplotlib','joblib']},'training_device':'CPU','parallel_grid_jobs':4}
    Path('outputs/environment.json').write_text(json.dumps(setup,indent=2))
    Path('requirements-lock.txt').write_text('\n'.join(f'{p}=={v}' for p,v in setup['packages'].items())+'\n')
    winner=val.iloc[0]
    notes=f'''# Review 2 experimental setup

This is supervised binary classification of recorded CVD status (0/1) in the primary dataset, not a validated forecast of future events. The Cleveland outcome is disease presence; Framingham predicts ten-year CHD. Their scores are separate experiments with different endpoints and features.

Primary data: Kaggle cardiovascular disease dataset, 70,000 raw rows. Final count: {audit['clean_records']}. Details of exclusions and class counts: preprocessing_audit.json. No resampling or class weighting is used on the primary data. Class distribution before and after balancing is therefore unchanged.

11 raw predictors: age, gender, height, weight, systolic BP, diastolic BP, cholesterol, glucose category, smoking, alcohol, activity. ID is excluded. Age is converted from days to years using 365.25 without rounding. Three deterministic derived predictors give 14 total: BMI = kg/(height in metres)^2; pulse pressure = systolic minus diastolic BP; MAP = (systolic + 2×diastolic)/3. No feature selection is performed.

Random row-wise, outcome-stratified split: 70% TRAINING, 15% validation, 15% test (integer rounding applies). Seed/random_state = 42 for splits, shuffled stratified five-fold CV and stochastic estimators. No sequence, time window, device split or temporal split applies. ID is a record identifier, not a verified repeated-subject identifier: subject independence cannot be guaranteed. Exact repeated original predictor profiles are removed in full, including conflicting labels; cross-split profile overlap is asserted zero. This conservative policy can remove distinct people with identical measured profiles.

All median imputation and LR standardization are learned inside the training pipeline and each CV fold. Deterministic row cleaning and feature formulas do not estimate population statistics. Test and validation never fit preprocessing. No outcome-derived input is used.

Baseline: DummyClassifier(strategy=prior) and scaled Logistic Regression. Grid-search candidates: LR C=[0.01,0.1,1,10]; RF n_estimators=[100,200,300], max_depth=[5,10,15]; XGBoost learning_rate=[0.01,0.05,0.1], n_estimators=[100,200,300], max_depth=[3,5,7]. LR C is inverse regularization strength, not a random seed; selected through training CV. LR uses L2 regularization and lbfgs, max_iter=2000. RF uses bootstrap aggregation and Gini impurity. XGBoost uses binary logistic objective, log-loss evaluation, histogram trees and gradient boosting. LR optimizes regularized logistic loss. RF has no learning-rate parameter; XGBoost's learning rate is tuned. No deep learning or early stopping is used; tree depth, tree count and regularization constrain complexity.

Winner: {winner.model}; selected parameters: {winner.best_params}. Validation ROC-AUC: {winner.roc_auc:.4f}. Training CV ROC-AUC: {winner.cv_roc_auc_mean:.4f} ± {winner.cv_roc_auc_std:.4f} (population SD across five folds). These folds also selected hyperparameters: they are tuning estimates, not nested-CV unbiased estimates. Final test ROC-AUC: {test.roc_auc:.4f}. Full metrics and baseline improvements are in validation_results.csv. Classification threshold remains 0.5.

The test set was examined in the previous pipeline. This is a corrected rerun on the same source data, not a new never-seen confirmatory evaluation. No test-driven hyperparameter changes were made in this corrected pipeline. A new independent cohort is required for a fresh confirmatory claim.

External experiments use the selected algorithm and hyperparameters, retrained separately on each dataset's own predictors, with stratified 70/30 train/test and seed 42. Median imputation is fitted after splitting and inside external training CV. RF/LR use training-only balanced class weights on external datasets; XGBoost uses no weighting. No external hyperparameter search is performed. These are approach-replication experiments, not external validation of a fixed fitted primary model. Different endpoints, predictors and small Cleveland size limit direct comparison.

SHAP uses a reproducible 500-record test sample. XGBoost uses native exact path-dependent TreeSHAP with training tree-cover statistics and a verified additive reconstruction of raw margins; RF/LR use 100 training background records. The global plot is an explicit beeswarm. Units depend on the selected estimator; see shap_metadata.json. Explanations describe model associations, not causal or clinical effects.

Hardware/software measured from this execution machine: {setup['cpu']}, {setup['logical_cpus']} logical CPUs, {setup['ram_gib']} GiB RAM, {setup['os']}. CPU training, four concurrent grid-search jobs. Exact package/Python versions: environment.json and requirements-lock.txt.

Charts: roc_curve.png, confusion_matrix.png, shap_summary.png, class_distribution.png (four meaningful charts); model_comparison.png (baseline comparison); cross_dataset_comparison.png (separate dataset comparison). Architecture source: architecture.drawio. PPT and report have not been updated.

Additional checks are generated by 06_additional_validation.py: exploratory 80/10/10 split sensitivity with frozen primary-selected settings, plus Cleveland training-only 5-fold CV repeated 10 times. Results and limitations: additional_validation/README.md. These supplement the main experiment; they do not change its official split or winner.
'''
    Path('outputs/review2_setup.md').write_text(notes)
    print(notes)

if __name__=='__main__': main()
