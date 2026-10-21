"""Exploratory 80/10/10 sensitivity check and Cleveland repeated training CV.

Primary 70/15/15 outputs remain unchanged. Hyperparameters are frozen from the
main experiment: repartitioning reuses previously inspected records, so the
80/10/10 scores are exploratory, not independent confirmation.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from pipeline_common import FEATURES, RAW_FEATURES, SEED, make_pipeline, evaluate

OUT = Path('outputs/additional_validation')

def split_check(selected):
    # Recreate original source order: main split files contain exactly the cleaned cohort.
    # Sort original raw records by original row index through the same cleaning rules,
    # rather than depending on the order of an earlier random partition.
    raw = pd.read_csv('data/cardio_train.csv', sep=';').drop(columns='id')
    df = raw.loc[~raw.duplicated(subset=RAW_FEATURES, keep=False)].copy()
    df = df.loc[df.ap_hi.gt(60)&df.ap_hi.lt(240)&df.ap_lo.gt(40)&df.ap_lo.lt(200)&df.ap_hi.ge(df.ap_lo)&df.height.gt(100)&df.height.lt(220)&df.weight.gt(30)&df.weight.lt(200)].copy()
    from pipeline_common import derive
    df['age'] = df.age / 365.25
    df = derive(df)
    train_val,test = train_test_split(df,test_size=.1,stratify=df.cardio,random_state=SEED)
    train,val = train_test_split(train_val,test_size=.1/.9,stratify=train_val.cardio,random_state=SEED)
    parts = {'train':train,'val':val,'test':test}
    keys = {s:set(map(tuple,d[RAW_FEATURES].to_numpy())) for s,d in parts.items()}
    for a,b in [('train','val'),('train','test'),('val','test')]: assert not keys[a]&keys[b]
    assert sum(map(len,parts.values())) == len(df)
    audit = {'ratio':'80/10/10','random_state':SEED,'records':len(df),'cross_split_duplicate_profiles':0,'preprocessing':'Median imputation fitted only on this 80% training subset; LR scaling also fitted only on training','selection':'Frozen primary-selected algorithm and hyperparameters; no supplementary selection or tuning','interpretation':'Exploratory sensitivity check; records and tuned settings reused from the main experiment, not independent confirmation','splits':{s:{'n':len(d),'fraction':len(d)/len(df),'class_counts':d.cardio.value_counts().to_dict()} for s,d in parts.items()}}
    (OUT/'split_80_10_10_audit.json').write_text(json.dumps(audit,indent=2))
    rows=[]
    for algorithm in ['Dummy',selected.model]:
        pipe=make_pipeline(algorithm,json.loads(selected.best_params) if algorithm==selected.model else None)
        pipe.fit(train[FEATURES],train.cardio)
        np.testing.assert_allclose(pipe.named_steps['imputer'].statistics_,train[FEATURES].median())
        for split,part in [('val',val),('test',test)]:
            rows.append({'model':algorithm,'split':split,**evaluate(pipe,part[FEATURES],part.cardio)})
        joblib.dump(pipe,OUT/f'split_80_10_10_{algorithm}.joblib')
    pd.DataFrame(rows).to_csv(OUT/'split_80_10_10_results.csv',index=False)
    print(json.dumps(audit,indent=2),flush=True)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

def repeated_cleveland(selected):
    df=pd.read_csv('data/cleveland.csv',na_values=['?'])
    predictors=df.drop(columns='target').columns.tolist()
    df=df.loc[~df.duplicated(subset=predictors,keep=False)]
    X=df[predictors].apply(pd.to_numeric,errors='raise'); y=(pd.to_numeric(df.target)>0).astype(int)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,stratify=y,random_state=SEED)
    cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=SEED)
    folds=list(cv.split(Xtr,ytr))
    for tr,va in folds:
        assert not set(tr)&set(va)
        assert set(ytr.iloc[tr])==set(ytr.iloc[va])=={0,1}
    all_rows=[]; summary=[]
    for algorithm in ['Dummy',selected.model]:
        pipe=make_pipeline(algorithm,json.loads(selected.best_params) if algorithm==selected.model else None,balanced=algorithm==selected.model)
        scores=cross_validate(pipe,Xtr,ytr,cv=folds,scoring={'roc_auc':'roc_auc','accuracy':'accuracy','recall':'recall','f1':'f1'},n_jobs=1,error_score='raise',return_estimator=True)
        for i,(tr,va) in enumerate(folds):
            # Evidence that each fold's imputer uses its own training records only.
            np.testing.assert_allclose(scores['estimator'][i].named_steps['imputer'].statistics_,Xtr.iloc[tr].median())
            all_rows.append({'model':algorithm,'repeat':i//5+1,'fold':i%5+1,'n_train':len(tr),'n_validation':len(va),**{m:float(scores['test_'+m][i]) for m in ['roc_auc','accuracy','recall','f1']}})
        summary.append({'model':algorithm,**{f'{m}_{stat}':float(getattr(np,stat)(scores['test_'+m])) for m in ['roc_auc','accuracy','recall','f1'] for stat in ['mean','std']}})
    pd.DataFrame(all_rows).to_csv(OUT/'cleveland_repeated_cv_folds.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'cleveland_repeated_cv_summary.csv',index=False)
    metadata={'strategy':'RepeatedStratifiedKFold','n_splits':5,'n_repeats':10,'fits_per_model':50,'random_state':SEED,'training_records':len(Xtr),'untouched_holdout_records':len(Xte),'holdout_evaluated_by_this_script':False,'hyperparameters':'Frozen from primary selection; no Cleveland tuning','preprocessing':'Median imputation fitted separately within each fold; verified against fold training medians','sd':'Population SD across 50 dependent fold scores, not a standard error or confidence interval'}
    (OUT/'cleveland_repeated_cv_audit.json').write_text(json.dumps(metadata,indent=2))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    return audit_note(metadata,summary)

def audit_note(metadata,summary):
    winner=summary[1]
    return f"""# Additional Review 2 validation

The main 70/15/15 experiment and selected model are unchanged.

## 80/10/10 check

Same cleaning and 14 predictors; stratified seed-42 random row split, 80% training and 10% each validation/test. Exact counts and class distributions: split_80_10_10_audit.json. Separate baseline and frozen-winner metrics: split_80_10_10_results.csv. No duplicate predictor profiles overlap. Pipelines are fitted afresh on the 80% training subset only.

This is an exploratory sensitivity check using previously inspected source records and parameters selected in the main experiment. It is not a new independent confirmation, and it is not used to choose the official split or model. Subject independence remains unverifiable.

## Repeated Cleveland CV

5 stratified folds × 10 repeats = 50 fits per model, seed 42, restricted to the original Cleveland training partition ({metadata['training_records']} records). The original {metadata['untouched_holdout_records']}-record holdout is excluded. The primary-selected algorithm/parameters are fixed; no external tuning occurs. Every fold fits its own imputer (and scaler for LR), with fitted medians checked against that fold's training data. Both Dummy and selected-model results are saved.

Selected-model ROC-AUC: {winner['roc_auc_mean']:.4f} ± {winner['roc_auc_std']:.4f}. SD is across 50 dependent fold scores, not a confidence interval or standard error. Individual repeats/folds and other metrics are in cleveland_repeated_cv_folds.csv. This supplements the existing single five-fold CV result; it does not replace or pool holdout metrics. Cleveland remains a separately retrained approach-replication experiment.
"""

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    selected=pd.read_csv('outputs/validation_results.csv').sort_values('roc_auc',ascending=False).iloc[0]
    split_check(selected)
    note=repeated_cleveland(selected)
    (OUT/'README.md').write_text(note)

if __name__=='__main__': main()
