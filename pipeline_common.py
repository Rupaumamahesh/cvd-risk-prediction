"""Shared, train-fitted preprocessing and reproducible model definitions."""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

SEED = 42
RAW_FEATURES = ['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']
FEATURES = RAW_FEATURES + ['bmi','pulse_pressure','map']

def derive(df):
    df = df.copy()
    df['bmi'] = df.weight / (df.height / 100)**2
    df['pulse_pressure'] = df.ap_hi - df.ap_lo
    df['map'] = (df.ap_hi + 2 * df.ap_lo) / 3
    return df

def make_pipeline(name, params=None, balanced=False):
    weight = 'balanced' if balanced else None
    models = {
        'Dummy': lambda: DummyClassifier(strategy='prior'),
        'LogisticRegression': lambda: LogisticRegression(max_iter=2000, random_state=SEED, class_weight=weight),
        'RandomForest': lambda: RandomForestClassifier(random_state=SEED, n_jobs=1, class_weight=weight),
        'XGBoost': lambda: XGBClassifier(random_state=SEED, n_jobs=1, eval_metric='logloss', tree_method='hist'),
    }
    pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler() if name == 'LogisticRegression' else 'passthrough'), ('model', models[name]())])
    if params:
        pipe.set_params(**{k if '__' in k else 'model__'+k: v for k,v in params.items()})
    return pipe

def evaluate(model, X, y):
    pred = model.predict(X)
    return dict(accuracy=accuracy_score(y,pred), precision=precision_score(y,pred,zero_division=0), recall=recall_score(y,pred,zero_division=0), f1=f1_score(y,pred,zero_division=0), roc_auc=roc_auc_score(y,model.predict_proba(X)[:,1]))
