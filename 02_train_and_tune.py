"""Training-only grid search; validation selects winner; no test access."""
import json
import os
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from pipeline_common import FEATURES, SEED, make_pipeline, evaluate

PARAM_GRIDS = {
 'Dummy': {},
 'LogisticRegression': {'model__C':[.01,.1,1,10]},
 'RandomForest': {'model__n_estimators':[100,200,300], 'model__max_depth':[5,10,15]},
 'XGBoost': {'model__learning_rate':[.01,.05,.1], 'model__n_estimators':[100,200,300], 'model__max_depth':[3,5,7]},
}

def main():
    Path('models').mkdir(exist_ok=True)
    train,val = [pd.read_csv(f'data/primary_{s}.csv') for s in ['train','val']]
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    results=[]
    for name, grid in PARAM_GRIDS.items():
        print('Tuning',name,flush=True)
        search=GridSearchCV(make_pipeline(name),grid,scoring={'roc_auc':'roc_auc','accuracy':'accuracy','f1':'f1'},refit='roc_auc',cv=cv,n_jobs=int(os.environ.get('CVD_JOBS','4')),error_score='raise')
        search.fit(train[FEATURES],train.cardio)
        pd.DataFrame(search.cv_results_).to_csv(f'outputs/cv_{name}.csv',index=False)
        row={'model':name,**evaluate(search.best_estimator_,val[FEATURES],val.cardio),'best_params':json.dumps(search.best_params_)}
        i=search.best_index_
        for metric in ['roc_auc','accuracy','f1']:
            row[f'cv_{metric}_mean']=search.cv_results_[f'mean_test_{metric}'][i]
            row[f'cv_{metric}_std']=search.cv_results_[f'std_test_{metric}'][i]
        results.append(row)
        joblib.dump(search.best_estimator_,f'models/{name}.joblib')
        print(row,flush=True)
    out=pd.DataFrame(results).sort_values('roc_auc',ascending=False)
    out['roc_auc_improvement_over_dummy']=out.roc_auc-out.loc[out.model.eq('Dummy'),'roc_auc'].iloc[0]
    out.to_csv('outputs/validation_results.csv',index=False)
    winner=out.iloc[0].model
    joblib.dump(joblib.load(f'models/{winner}.joblib'),'models/best_model.joblib')
    joblib.dump(FEATURES,'models/feature_list.joblib')
    Path('outputs/selection.json').write_text(json.dumps({'winner':winner,'selection':'highest validation ROC-AUC','threshold':.5,'test_used_for_selection':False},indent=2))
    print(out.to_string(index=False),flush=True)

if __name__=='__main__': main()
