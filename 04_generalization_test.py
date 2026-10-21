"""Separate within-dataset evaluations of the selected approach, not fixed-model transfer."""
import json
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from pipeline_common import SEED, make_pipeline, evaluate

def main():
    selected=pd.read_csv('outputs/validation_results.csv').sort_values('roc_auc',ascending=False).iloc[0]
    name=selected.model; params=json.loads(selected.best_params)
    results=[]; audits={}
    for dataset,path,target in [('Cleveland','data/cleveland.csv','target'),('Framingham','data/framingham.csv','TenYearCHD')]:
        df=pd.read_csv(path,na_values=['?'])
        predictors=list(df.drop(columns=target).columns)
        raw=len(df)
        df=df.loc[~df.duplicated(subset=predictors,keep=False)].copy()
        y=pd.to_numeric(df[target]); X=df[predictors].apply(pd.to_numeric,errors='raise')
        if dataset=='Cleveland': y=(y>0).astype(int)
        assert set(y.unique())=={0,1}
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,stratify=y,random_state=SEED)
        assert not set(map(tuple,Xtr.fillna(-999999).to_numpy())) & set(map(tuple,Xte.fillna(-999999).to_numpy()))
        audits[dataset]={'raw_records':raw,'clean_records':len(df),'train_records':len(Xtr),'test_records':len(Xte),'class_counts':y.value_counts().to_dict(),'train_class_counts':ytr.value_counts().to_dict(),'test_class_counts':yte.value_counts().to_dict(),'train_missing_cells':int(Xtr.isna().sum().sum()),'test_missing_cells':int(Xte.isna().sum().sum()),'features':predictors,'imputation':'median fitted inside training pipeline and each CV fold','balancing':'No resampling; class_weight balanced for RF/LR; no weighting for XGBoost','evaluation':'70/30 stratified holdout; no external tuning; training-only 5-fold CV'}
        for algorithm in ['Dummy',name]:
            pipe=make_pipeline(algorithm,params if algorithm==name else None,balanced=algorithm==name)
            cv=cross_validate(pipe,Xtr,ytr,scoring='roc_auc',cv=StratifiedKFold(5,shuffle=True,random_state=SEED),n_jobs=1)
            pipe.fit(Xtr,ytr)
            row={'dataset':dataset,'model':algorithm,**evaluate(pipe,Xte,yte),'n_records':len(df),'n_features':X.shape[1],'cv_roc_auc_mean':cv['test_score'].mean(),'cv_roc_auc_std':cv['test_score'].std()}
            results.append(row)
            if algorithm==name:
                pd.DataFrame({'actual':yte,'probability':pipe.predict_proba(Xte)[:,1],'prediction':pipe.predict(Xte)}).to_csv(f'outputs/{dataset.lower()}_predictions.csv',index=False)
            print(row,flush=True)
    pd.DataFrame(results).to_csv('outputs/generalization_results.csv',index=False)
    with open('outputs/external_audit.json','w') as f: json.dump(audits,f,indent=2)

if __name__=='__main__': main()
