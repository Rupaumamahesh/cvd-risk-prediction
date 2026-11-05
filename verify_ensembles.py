"""Verify saved ensembles, fold coverage, arithmetic, metrics and artifact preservation."""
import importlib
import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from ensemble_models import assert_training_preprocessing
from pipeline_common import FEATURES

m=importlib.import_module('08_train_ensembles')
OUT=m.OUT
folds=pd.read_csv(OUT/'outer_cv_folds.csv')
summary=pd.read_csv(OUT/'outer_cv_summary.csv')
oof=pd.read_csv(OUT/'outer_oof_predictions.csv')
assert len(oof)==47968 and oof.outer_fold.nunique()==5
assert set(oof.actual)=={0,1} and not oof.isna().any().any()
for name,g in folds.groupby('model'):
    assert len(g)==5 and g.fold.nunique()==5
    r=summary.loc[summary.model.eq(name)].iloc[0]
    for metric in ['roc_auc','average_precision','accuracy','precision','recall','f1','specificity','npv','log_loss','brier_score']:
        np.testing.assert_allclose([g[metric].mean(),g[metric].std(ddof=0)],[r[metric+'_mean'],r[metric+'_std']])
    for _,fold in g.iterrows():
        part=oof.loc[oof.outer_fold.eq(fold.fold)]
        assert len(part)==fold.n_heldout
        actual=m.evaluate_probabilities(part.actual,part[name])
        for metric,value in actual.items(): np.testing.assert_allclose(value,fold[metric])
print('PASS: outer OOF coverage, five-fold metrics, means and standard deviations')
inner=pd.read_csv(OUT/'inner_fold_audit.csv')
final=pd.read_csv(OUT/'final_inner_fold_audit.csv')
assert len(inner)==75 and len(final)==15
assert inner.disjoint.all() and inner.preprocessing_statistics_verified.all()
assert final.disjoint.all() and final.preprocessing_statistics_verified.all()
train=pd.read_csv('data/primary_train.csv')
val=pd.read_csv('data/primary_val.csv')
validation=pd.read_csv(OUT/'validation_results.csv')
with threadpool_limits(limits=1):
    for name,file in [('SoftVoting','soft_voting.joblib'),('Stacking','stacking.joblib')]:
        model=joblib.load(m.MODELS/file)
        for _,base in model.estimators:
            assert_training_preprocessing(base,train[FEATURES])
        # Verify numerical combination and serialization on held-out examples.
        sample=val[FEATURES].iloc[:100]
        base=model.member_probabilities(sample)
        expected=base.mean(axis=1) if model.combiner is None else model.combiner.predict_proba(base)[:,1]
        np.testing.assert_allclose(model.predict_proba(sample)[:,1],expected)
        p=pd.read_csv(OUT/f'validation_predictions_{name}.csv')
        np.testing.assert_allclose(model.predict_proba(sample)[:,1],p.probability.iloc[:100])
        row=validation.loc[validation.model.eq(name)].iloc[0]
        for metric,value in m.evaluate_probabilities(p.actual,p.probability).items(): np.testing.assert_allclose(value,row[metric])
        assert set(np.unique(model.predict(sample))).issubset({0,1})
        if name=='Stacking':
            training_oof=pd.read_csv(OUT/'base_training_oof_predictions.csv')
            features=training_oof[['XGBoost','RandomForest','KNN']].to_numpy()
            from sklearn.base import clone
            refit=clone(model.combiner).fit(features,train.cardio)
            np.testing.assert_allclose(refit.coef_,model.combiner.coef_,rtol=1e-6,atol=1e-7)
            np.testing.assert_allclose(refit.intercept_,model.combiner.intercept_,rtol=1e-6,atol=1e-7)
        print('PASS:',name,'preprocessing, serialized inference, combination arithmetic and validation metrics')
selection=json.loads((OUT/'selection.json').read_text())
assert selection['winner']==validation.sort_values('roc_auc',ascending=False,kind='stable').iloc[0].model
p=pd.read_csv(OUT/'selected_test_predictions.csv')
test=pd.read_csv(OUT/'selected_test_results.csv').iloc[0]
for metric,value in m.evaluate_probabilities(p.actual,p.probability).items(): np.testing.assert_allclose(value,test[metric])
verification=json.loads((OUT/'verification.json').read_text())
for path,expected in verification['protected_file_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
print('PASS: validation-based winner, saved test metrics and original artifacts preserved')
