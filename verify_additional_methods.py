"""Verify expanded-model artifacts and preprocessing leakage controls."""
import importlib
import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from threadpoolctl import threadpool_limits

m=importlib.import_module('07_compare_additional_methods')
train=pd.read_csv('data/primary_train.csv')
val=pd.read_csv('data/primary_val.csv')
rows=pd.read_csv(m.OUT/'new_methods_validation.csv')
for name in m.CONFIGS:
    model=joblib.load(m.MODELS/f'{name}.joblib')
    m.verify_preprocessing(model,train)
    cv=pd.read_csv(m.OUT/f'cv_{name}.csv')
    best=cv.loc[cv.rank_test_roc_auc.eq(1)].iloc[0]
    fold_scores=np.array([best[f'split{i}_test_roc_auc'] for i in range(5)])
    saved=rows.loc[rows.model.eq(name)].iloc[0]
    np.testing.assert_allclose([fold_scores.mean(),fold_scores.std()], [saved.cv_roc_auc_mean,saved.cv_roc_auc_std])
    predictions=pd.read_csv(m.OUT/f'validation_predictions_{name}.csv')
    np.testing.assert_allclose(m.roc_auc_score(predictions.actual,predictions.score),saved.roc_auc)
    # A held-out record with an unseen category cannot change fitted categories,
    # medians or scaling means; transformer must still handle it at inference.
    pre=model.named_steps['preprocess']
    before=joblib.hash(pre)
    sentinel=val[m.FEATURES].iloc[:2].copy()
    sentinel.loc[sentinel.index[0],m.NUMERIC]=np.nan
    sentinel.loc[sentinel.index[0],'gender']=999
    with threadpool_limits(limits=1): model.predict(sentinel)
    assert joblib.hash(pre)==before
    # Refit preprocessing alone on a small explicit fold; held-out extremes
    # and unknown categories must be absent from its fitted statistics.
    fold=train.iloc[:200]
    fold_pre=clone(pre).fit(fold[m.FEATURES],fold.cardio)
    np.testing.assert_allclose(fold_pre.named_transformers_['numeric']['imputer'].statistics_,fold[m.NUMERIC].median())
    np.testing.assert_allclose(fold_pre.named_transformers_['numeric']['scale'].mean_,fold[m.NUMERIC].mean())
    assert 999 not in fold_pre.named_transformers_['categorical']['encode'].categories_[0]
    print('PASS:',name,'CV summary, saved validation scores and preprocessing isolation')
verification=json.loads((m.OUT/'verification.json').read_text())
for path,expected in verification['protected_file_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
comparison=pd.read_csv(m.OUT/'all_methods_validation.csv')
assert len(comparison)==7 and comparison.model.nunique()==7
assert comparison.iloc[0].model==json.loads((m.OUT/'selection.json').read_text())['winner']
print('PASS: seven-model selection and original artifact preservation')
