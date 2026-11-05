"""Additional primary-data algorithms; preserve original model/results artifacts.

Tune three new algorithms on the existing training partition with 5-fold CV,
compare all algorithms on validation, then evaluate only the extended winner.
This is exploratory development on previously inspected source data.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cvd-matplotlib')
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, log_loss,
                             brier_score_loss, confusion_matrix, RocCurveDisplay)
from threadpoolctl import threadpool_limits
from plot_labels import label_cv, draw_roc
from pipeline_common import FEATURES, RAW_FEATURES, SEED

OUT = Path('outputs/additional_methods')
MODELS = Path('models/additional_methods')
NUMERIC = ['age','height','weight','ap_hi','ap_lo','bmi','pulse_pressure','map']
CATEGORICAL = [f for f in FEATURES if f not in NUMERIC]
CONFIGS = {
    'LinearSVM': (LinearSVC(dual=False, max_iter=20000, random_state=SEED),
                  {'model__C': [0.01, 0.1, 1.0]}),
    'KNN': (KNeighborsClassifier(algorithm='brute', n_jobs=1),
            {'model__n_neighbors': [15, 51, 101], 'model__weights': ['uniform','distance']}),
    'GaussianNB': (GaussianNB(), {'model__var_smoothing': [1e-9, 1e-6, 1e-3]}),
}

def pipeline(estimator):
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median')),
                        ('scale', StandardScaler())])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return Pipeline([('preprocess', ColumnTransformer([
        ('numeric', numeric, NUMERIC), ('categorical', categorical, CATEGORICAL)
    ])), ('model', estimator)])

def scores(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1], 'probability'
    return model.decision_function(X), 'decision_margin'

def metrics(model, X, y):
    pred = model.predict(X)
    score, kind = scores(model, X)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    result = dict(accuracy=accuracy_score(y,pred), precision=precision_score(y,pred,zero_division=0),
                  recall=recall_score(y,pred,zero_division=0), f1=f1_score(y,pred,zero_division=0),
                  roc_auc=roc_auc_score(y,score), average_precision=average_precision_score(y,score),
                  specificity=tn/(tn+fp), npv=tn/(tn+fn) if tn+fn else np.nan,
                  tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),score_type=kind)
    if kind == 'probability':
        result.update(log_loss=log_loss(y,score),brier_score=brier_score_loss(y,score))
    else:
        result.update(log_loss=np.nan,brier_score=np.nan)
    return result

def verify_preprocessing(model, train):
    pre = model.named_steps['preprocess']
    num = pre.named_transformers_['numeric']
    np.testing.assert_allclose(num['imputer'].statistics_, train[NUMERIC].median())
    transformed = num['imputer'].transform(train[NUMERIC])
    np.testing.assert_allclose(num['scale'].mean_, transformed.mean(axis=0))
    cat = pre.named_transformers_['categorical']
    for col, replacement in zip(CATEGORICAL, cat['imputer'].statistics_):
        assert replacement in train[col].mode().values

def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def plot_comparison(comparison):
    fig,ax = plt.subplots(figsize=(11,6))
    bars = ax.bar(comparison.model,comparison.roc_auc,color='#447b98',label='Validation ROC-AUC')
    ax.bar_label(bars,labels=[f'{v:.4f}' for v in comparison.roc_auc],padding=12,fontsize=10)
    label_cv(ax,comparison.cv_roc_auc_mean,comparison.cv_roc_auc_std)
    ax.errorbar(comparison.model,comparison.cv_roc_auc_mean,yerr=comparison.cv_roc_auc_std,fmt='ko',capsize=4,label='Training CV mean ± SD')
    ax.set(ylabel='ROC-AUC',ylim=(0,1),title='Additional algorithms: same primary training and validation partitions')
    ax.tick_params(axis='x',rotation=20);ax.legend();fig.tight_layout();fig.savefig(OUT/'model_comparison.png',dpi=180);plt.close(fig)

def plot_roc_curves(models,val):
    fig,ax = plt.subplots(figsize=(8,6))
    with threadpool_limits(limits=1):
        for name in models:
            score,_ = scores(models[name],val[FEATURES])
            draw_roc(ax,val.cardio,score,name)
    ax.plot([0,1],[0,1],'k--');ax.set_title('Validation ROC curves: expanded algorithm comparison')
    fig.tight_layout();fig.savefig(OUT/'validation_roc_curves.png',dpi=180);plt.close(fig)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    protected = ['models/best_model.joblib','outputs/validation_results.csv',
                 'outputs/test_results.csv','outputs/test_predictions.csv','outputs/selection.json']
    fingerprints = {p:fingerprint(p) for p in protected}
    train = pd.read_csv('data/primary_train.csv')
    val = pd.read_csv('data/primary_val.csv')
    tr_profiles = set(map(tuple, train[RAW_FEATURES].to_numpy()))
    assert not tr_profiles.intersection(map(tuple,val[RAW_FEATURES].to_numpy()))
    X,y = train[FEATURES],train.cardio
    folds = list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(X,y))
    rows = []; models = {}
    for name,(estimator,grid) in CONFIGS.items():
        print('Tuning',name,flush=True)
        search = GridSearchCV(pipeline(estimator), grid, cv=folds,
            scoring={'roc_auc':'roc_auc','accuracy':'accuracy','f1':'f1'}, refit='roc_auc',
            n_jobs=int(os.environ.get('CVD_JOBS','2')), error_score='raise')
        # Avoid nested BLAS parallelism on the four-logical-CPU machine.
        with joblib.parallel_backend('loky', inner_max_num_threads=1), threadpool_limits(limits=1):
            search.fit(X,y)
            result = metrics(search.best_estimator_,val[FEATURES],val.cardio)
        verify_preprocessing(search.best_estimator_, train)
        models[name] = search.best_estimator_
        joblib.dump(search.best_estimator_, MODELS/f'{name}.joblib')
        pd.DataFrame(search.cv_results_).to_csv(OUT/f'cv_{name}.csv',index=False)
        row = dict(model=name, **result, best_params=json.dumps(search.best_params_))
        for metric in ['roc_auc','accuracy','f1']:
            row[f'cv_{metric}_mean'] = search.cv_results_[f'mean_test_{metric}'][search.best_index_]
            row[f'cv_{metric}_std'] = search.cv_results_[f'std_test_{metric}'][search.best_index_]
        rows.append(row)
        score,kind = scores(search.best_estimator_,val[FEATURES])
        pd.DataFrame({'actual':val.cardio,'score':score,'prediction':search.best_estimator_.predict(val[FEATURES])}).to_csv(OUT/f'validation_predictions_{name}.csv',index=False)
        print(row,flush=True)
        pd.DataFrame(rows).to_csv(OUT/'new_methods_validation.csv',index=False)

    original = pd.read_csv('outputs/validation_results.csv')
    # Re-score original saved models on validation only for consistent extra metrics.
    for _,old in original.iterrows():
        fitted = joblib.load(f'models/{old.model}.joblib')
        result = metrics(fitted,val[FEATURES],val.cardio)
        np.testing.assert_allclose(result['roc_auc'],old.roc_auc)
        row = old.to_dict(); row.update(result)
        rows.append(row); models[old.model] = fitted
    comparison = pd.DataFrame(rows).sort_values('roc_auc',ascending=False,kind='stable')
    comparison['roc_auc_improvement_over_dummy'] = comparison.roc_auc - comparison.loc[comparison.model.eq('Dummy'),'roc_auc'].iloc[0]
    comparison.to_csv(OUT/'all_methods_validation.csv',index=False)
    winner = comparison.iloc[0].model
    selection = {'winner':winner,'rule':'highest primary validation ROC-AUC across all seven candidates',
        'test_used_for_selection':False,'original_artifacts_preserved':True,
        'scope':'Exploratory extension; same previously inspected source data and validation set reused',
        'grids':{n:g for n,(_,g) in CONFIGS.items()},'seed':SEED,'cv_folds':5,
        'numeric_features':NUMERIC,'categorical_features':CATEGORICAL,
        'svm_output':'Decision margin, threshold zero; not probability; no log loss or Brier score',
        'other_outputs':'Probabilities; native predict class rule (approximately threshold 0.5)',
        'gaussian_nb_limitations':'Conditional independence violated by derived BP/BMI inputs; Gaussian approximation imperfect for one-hot binary columns',
        'comparison_note':'Same raw/derived inputs and folds; new methods use categorical encoding, old pipelines unchanged; comparison includes algorithm-specific preprocessing'}
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2))
    joblib.dump(models[winner],MODELS/'extended_winner.joblib')
    # Freeze selection before any test file is read.
    if winner == json.loads(Path('outputs/selection.json').read_text())['winner']:
        test_predictions = pd.read_csv('outputs/test_predictions.csv')
        score = test_predictions.probability
        actual = test_predictions.actual
        test_result = pd.read_csv('outputs/test_results.csv').iloc[0].to_dict()
        tn,fp,fn,tp = confusion_matrix(actual,test_predictions.prediction,labels=[0,1]).ravel()
        test_result.update(average_precision=average_precision_score(actual,score),
            log_loss=log_loss(actual,score),brier_score=brier_score_loss(actual,score),
            specificity=tn/(tn+fp),npv=tn/(tn+fn),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),score_type='probability')
        test_source = 'Reused original winner test predictions; no new test inference'
    else:
        test = pd.read_csv('data/primary_test.csv')
        assert not tr_profiles.intersection(map(tuple,test[RAW_FEATURES].to_numpy()))
        test_result = metrics(models[winner],test[FEATURES],test.cardio)
        score,kind = scores(models[winner],test[FEATURES])
        pd.DataFrame({'actual':test.cardio,'score':score,'prediction':models[winner].predict(test[FEATURES])}).to_csv(OUT/'extended_winner_test_predictions.csv',index=False)
        test_source = 'Exploratory evaluation of frozen extended winner on previously inspected test records'
    pd.DataFrame([dict(model=winner,**test_result,source=test_source)]).to_csv(OUT/'extended_winner_test_results.csv',index=False)
    plot_comparison(comparison)
    plot_roc_curves({name:models[name] for name in comparison.model},val)
    assert all(fingerprint(p)==h for p,h in fingerprints.items())
    (OUT/'verification.json').write_text(json.dumps({'passed':True,'protected_file_sha256':fingerprints,
        'checks':['Training/validation exact profile isolation','Training-only medians and scaler means for new fitted pipelines',
                  'CV fold indices fixed and reused across new methods','Original validation AUC reproduced',
                  'Winner frozen before test access','Original model and result fingerprints unchanged']},indent=2))
    notes = f'''# Additional machine-learning methods

Implemented LinearSVM, KNN and GaussianNB alongside the original Dummy, LogisticRegression, RandomForest and XGBoost. Main 70/15/15 split, 14 source predictors, seed 42 and stratified five-fold training CV are retained. No external datasets are used in this extension. No resampling, class weights, early stopping, calibration or ensemble-of-different-algorithms is added.

Numeric inputs use training-fitted median imputation and standardization. Categorical inputs use training-fitted most-frequent imputation and one-hot encoding with unknown categories ignored. Every candidate's preprocessing is inside its CV pipeline. This is an algorithm-and-preprocessing comparison: original models retain their original preprocessing, while new algorithms use categorical encoding.

Linear SVM uses squared hinge loss, L2 regularization, a linear boundary and maximum 20,000 iterations. C=[0.01,0.1,1] is tuned. Decision margins are valid ranking scores for ROC-AUC/average precision but are not probabilities; classification uses a zero margin and probability losses are omitted. A kernel SVM is omitted to control runtime on 47,968 training records.

KNN uses brute-force Euclidean distances, k=[15,51,101] and uniform/distance voting. It is sensitive to scales, categorical representation and correlated predictors; prediction retains and searches training records, so its saved artifact and inference cost can be larger.

GaussianNB tunes var_smoothing=[1e-9,1e-6,1e-3]. It is an inexpensive probabilistic reference, not a well-specified clinical generative model: correlated/derived features violate conditional independence, and one-hot columns are not Gaussian. Predicted probabilities should not be assumed calibrated.

Extended validation winner: {winner}. Its test result is exploratory. {test_source}. The primary model/results and previous report remain unchanged. Repeated use of validation and previously inspected test records does not yield independent confirmation. Training CV mean ± population SD reflects tuning folds, not nested-CV uncertainty or confidence intervals. No claim of statistical superiority, subject independence or clinical readiness is made.

Outputs: all_methods_validation.csv (seven models, discrimination, classification and probability metrics), new_methods_validation.csv, cv_*.csv, validation_predictions_*.csv, model_comparison.png, validation_roc_curves.png, extended_winner_test_results.csv, selection.json and verification.json. Fitted pipelines: models/additional_methods/.

Reproduce: venv/bin/python 07_compare_additional_methods.py (CVD_JOBS defaults to 2). The original training script remains unchanged.

Method references:
- https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
- https://scikit-learn.org/stable/modules/neighbors.html
- https://scikit-learn.org/stable/modules/naive_bayes.html
'''
    (OUT/'README.md').write_text(notes)
    print('\nFINAL VALIDATION COMPARISON\n'+comparison[['model','roc_auc','cv_roc_auc_mean','cv_roc_auc_std','recall','average_precision']].to_string(index=False),flush=True)
    print('Extended winner:',winner,'\n',test_result,flush=True)

if __name__ == '__main__': main()
