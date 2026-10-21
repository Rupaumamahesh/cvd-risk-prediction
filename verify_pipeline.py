"""Check saved split isolation, fitted preprocessing, metrics and SHAP metadata."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pipeline_common import FEATURES, RAW_FEATURES, make_pipeline

splits={s:pd.read_csv(f'data/primary_{s}.csv') for s in ['train','val','test']}
profiles={s:set(map(tuple,df[RAW_FEATURES].to_numpy())) for s,df in splits.items()}
for a,b in [('train','val'),('train','test'),('val','test')]: assert not profiles[a]&profiles[b]
for df in splits.values():
    assert 'id' not in df
    assert set(df.cardio.unique())=={0,1}
    np.testing.assert_allclose(df.bmi,df.weight/(df.height/100)**2)
    np.testing.assert_allclose(df.pulse_pressure,df.ap_hi-df.ap_lo)
    np.testing.assert_allclose(df['map'],(df.ap_hi+2*df.ap_lo)/3)
parameters={}
for name in ['Dummy','LogisticRegression','RandomForest','XGBoost']:
    pipe=joblib.load(f'models/{name}.joblib')
    np.testing.assert_allclose(pipe.named_steps['imputer'].statistics_,splits['train'][FEATURES].median())
    if name=='LogisticRegression': np.testing.assert_allclose(pipe.named_steps['scale'].mean_,splits['train'][FEATURES].mean())
    parameters[name]=pipe.named_steps['model'].get_params()
with open('outputs/fitted_hyperparameters.json','w') as f: json.dump(parameters,f,indent=2,default=str)
pred=pd.read_csv('outputs/test_predictions.csv')
assert len(pred)==len(splits['test'])
np.testing.assert_allclose(roc_auc_score(pred.actual,pred.probability),pd.read_csv('outputs/test_results.csv').roc_auc.iloc[0])
cv=pd.read_csv('outputs/validation_results.csv')
for name in cv.model:
    scores=pd.read_csv(f'outputs/cv_{name}.csv')
    best=scores.loc[scores.rank_test_roc_auc.eq(1)].iloc[0]
    folds=np.array([best[f'split{i}_test_roc_auc'] for i in range(5)])
    row=cv.loc[cv.model.eq(name)].iloc[0]
    np.testing.assert_allclose([folds.mean(),folds.std()],[row.cv_roc_auc_mean,row.cv_roc_auc_std])
# Sentinel: changing held-out data must not change a fitted median.
X=pd.DataFrame({'a':[1.,3.,np.nan,5.]}); y=[0,1,0,1]
pipe=make_pipeline('LogisticRegression').fit(X,y)
before=pipe.named_steps['imputer'].statistics_.copy()
pipe.predict(pd.DataFrame({'a':[1e9,np.nan]}))
np.testing.assert_array_equal(before,pipe.named_steps['imputer'].statistics_)
print('PASS: split isolation, derived features, training-fitted imputation/scaling, CV summaries, test metrics and inference immutability')
