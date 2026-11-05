"""Leakage-controlled soft voting and OOF stacking; preserve original artifacts.

Base settings frozen from earlier primary training searches. Five outer folds
assess the entire inner-OOF stacking procedure and tune combiner C. These are
exploratory tuning estimates, not fully nested base-hyperparameter selection.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/cvd-matplotlib')
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix, log_loss, brier_score_loss,
    RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.calibration import calibration_curve
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline_common import FEATURES, RAW_FEATURES, SEED
from ensemble_models import FittedProbabilityEnsemble, assert_training_preprocessing
from plot_labels import draw_roc, draw_pr

OUT = Path('outputs/ensembles')
MODELS = Path('models/ensembles')
CS = [0.01,0.1,1.0]
MEMBERS = [('XGBoost','models/XGBoost.joblib'),
           ('RandomForest','models/RandomForest.joblib'),
           ('KNN','models/additional_methods/KNN.joblib')]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def evaluate_probabilities(y,p):
    p=np.asarray(p); pred=(p>=0.5).astype(int)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(roc_auc=roc_auc_score(y,p),average_precision=average_precision_score(y,p),
        accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),
        recall=recall_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),
        specificity=tn/(tn+fp),npv=tn/(tn+fn) if tn+fn else np.nan,
        log_loss=log_loss(y,p),brier_score=brier_score_loss(y,p),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def fit_member(name, prototype, X, y, Xheld, splits):
    """Each OOF record is predicted only by a model excluding it from fitting."""
    with threadpool_limits(limits=1):
        oof=np.full(len(X),np.nan); coverage=np.zeros(len(X),dtype=int)
        audits=[]
        for fold,(tr,va) in enumerate(splits,1):
            assert not set(tr)&set(va)
            assert len(tr)+len(va)==len(X)
            assert set(y.iloc[tr])==set(y.iloc[va])=={0,1}
            model=clone(prototype).fit(X.iloc[tr],y.iloc[tr])
            assert_training_preprocessing(model,X.iloc[tr])
            oof[va]=model.predict_proba(X.iloc[va])[:,1]
            coverage[va]+=1
            audits.append(dict(member=name,inner_fold=fold,n_fit=len(tr),n_oof=len(va),
                disjoint=True,preprocessing_statistics_verified=True))
        assert (coverage==1).all() and np.isfinite(oof).all()
        fitted=clone(prototype).fit(X,y)
        assert_training_preprocessing(fitted,X)
        held=fitted.predict_proba(Xheld)[:,1]
    return name,oof,held,fitted,audits

def fit_members(prototypes,X,y,Xheld):
    splits=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(X,y))
    with joblib.parallel_backend('loky',inner_max_num_threads=1):
        results=Parallel(n_jobs=int(os.environ.get('CVD_JOBS','2')))(
            delayed(fit_member)(name,model,X,y,Xheld,splits) for name,model in prototypes)
    return (np.column_stack([r[1] for r in results]),np.column_stack([r[2] for r in results]),
            [(r[0],r[3]) for r in results],[a for r in results for a in r[4]])

def combiner(C):
    return LogisticRegression(C=C,solver='lbfgs',max_iter=2000,random_state=SEED)

def make_plots(validation, probabilities, y):
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,metric,title in [(axes[0],'roc_auc','Validation ROC-AUC'),(axes[1],'recall','Validation recall at threshold 0.5')]:
        bars=ax.bar(validation.model,validation[metric],color=['#447b98','#659d78','#ad7b47'])
        ax.bar_label(bars,labels=[f'{v:.4f}' for v in validation[metric]],padding=4)
        ax.set(ylim=(0,1.05),title=title); ax.tick_params(axis='x',rotation=15)
    fig.suptitle('Individual XGBoost versus heterogeneous ensembles')
    fig.tight_layout();fig.savefig(OUT/'ensemble_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for name,p in probabilities.items():
        draw_roc(axes[0],y,p,name)
        draw_pr(axes[1],y,p,name)
        observed,predicted=calibration_curve(y,p,n_bins=10,strategy='quantile')
        axes[2].plot(predicted,observed,'o-',label=f'{name} (Brier={brier_score_loss(y,p):.4f})')
    axes[0].plot([0,1],[0,1],'k--');axes[0].set_title('Validation ROC')
    axes[1].axhline(np.mean(y),color='black',linestyle='--');axes[1].set_title('Validation precision–recall')
    axes[2].plot([0,1],[0,1],'k--');axes[2].set(xlabel='Mean predicted probability',ylabel='Observed positive fraction',title='Validation calibration',xlim=(0,1),ylim=(0,1));axes[2].legend()
    fig.tight_layout();fig.savefig(OUT/'validation_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,len(probabilities),figsize=(18,9))
    table=[]
    for ax,(name,p) in zip(axes,probabilities.items()):
        observed,predicted=calibration_curve(y,p,n_bins=10,strategy='quantile')
        ax.plot([0,1],[0,1],'k--',alpha=.5);ax.plot(predicted,observed,'o-')
        for i,(xv,yv) in enumerate(zip(predicted,observed)):
            ax.annotate(str(i+1),(xv,yv),xytext=(3,10 if i%2==0 else -12),
                        textcoords='offset points',fontsize=8)
            table.append(dict(model=name,bin=i+1,mean_predicted_probability=xv,observed_positive_fraction=yv))
        cells=[[str(i+1),f'{x:.3f}',f'{v:.3f}'] for i,(x,v) in enumerate(zip(predicted,observed))]
        numeric_table=ax.table(cellText=cells,colLabels=['Bin','Predicted mean','Observed fraction'],cellLoc='center',bbox=[0,-1.42,1,1.12])
        numeric_table.auto_set_font_size(False);numeric_table.set_fontsize(9)
        ax.set(title=f'{name}: calibration bins',xlabel='Mean predicted probability',ylabel='Observed positive fraction',xlim=(0,1),ylim=(0,1))
    fig.suptitle('Point numbers match the numeric bin tables below',fontsize=13)
    fig.subplots_adjust(bottom=.56,top=.91,wspace=.30)
    fig.savefig(OUT/'calibration_numeric.png',dpi=200);plt.close(fig)
    pd.DataFrame(table).to_csv(OUT/'calibration_bin_values.csv',index=False)


def main():
    OUT.mkdir(parents=True,exist_ok=True);MODELS.mkdir(parents=True,exist_ok=True)
    protected=[Path('models/best_model.joblib'),Path('outputs/test_results.csv'),Path('outputs/test_predictions.csv'),
        Path('outputs/validation_results.csv'),Path('outputs/selection.json'),Path('outputs/additional_methods/all_methods_validation.csv')]
    for directory in ['models','models/additional_methods']:
        protected.extend(p for p in Path(directory).glob('*.joblib') if p not in protected)
    before={str(p):digest(p) for p in protected}
    train=pd.read_csv('data/primary_train.csv');val=pd.read_csv('data/primary_val.csv')
    profiles=lambda d:set(map(tuple,d[RAW_FEATURES].to_numpy()))
    assert not profiles(train)&profiles(val)
    X,y=train[FEATURES],train.cardio
    prototypes=[(name,clone(joblib.load(path))) for name,path in MEMBERS]
    config={'members':{name:model.get_params(deep=True) for name,model in prototypes},
        'combiner_C_grid':CS,'outer_folds':5,'inner_oof_folds':5,'seed':SEED,
        'base_hyperparameters':'Frozen from previous training searches; not retuned inside outer folds',
        'interpretation':'Exploratory tuning estimates: outer folds select combiner C; historical base tuning used this same training partition',
        'preprocessing':'Each base learner retains its own pipeline; fitted inside every inner/outer training subset',
        'meta_features':'Three class-1 probabilities only; no original input passthrough',
        'calibration':'Diagnostic curves and probability scores; no post-hoc calibration or threshold tuning',
        'threshold':0.5}
    (OUT/'config.json').write_text(json.dumps(config,indent=2,default=str))
    candidates=['XGBoost','SoftVoting']+[f'Stacking_C={C:g}' for C in CS]
    outer_predictions={name:np.full(len(X),np.nan) for name in candidates}
    outer_fold=np.zeros(len(X),dtype=int);fold_rows=[];audit_rows=[]
    for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED).split(X,y),1):
        print(f'Outer fold {fold}/5: generating inner out-of-fold predictions and fitting base pipelines',flush=True)
        assert not set(tr)&set(va)
        assert not profiles(train.iloc[tr])&profiles(train.iloc[va])
        base_oof,base_held,_,audits=fit_members(prototypes,X.iloc[tr],y.iloc[tr],X.iloc[va])
        audit_rows.extend([dict(outer_fold=fold,**a) for a in audits])
        probabilities={'XGBoost':base_held[:,0],'SoftVoting':base_held.mean(axis=1)}
        for C in CS:
            meta=combiner(C).fit(base_oof,y.iloc[tr])
            probabilities[f'Stacking_C={C:g}']=meta.predict_proba(base_held)[:,1]
        outer_fold[va]=fold
        for name,p in probabilities.items():
            outer_predictions[name][va]=p
            fold_rows.append(dict(model=name,fold=fold,n_fit=len(tr),n_heldout=len(va),**evaluate_probabilities(y.iloc[va],p)))
        pd.DataFrame(fold_rows).to_csv(OUT/'outer_cv_folds.csv',index=False)
        pd.DataFrame(audit_rows).to_csv(OUT/'inner_fold_audit.csv',index=False)
        print('Fold complete:',{name:round(roc_auc_score(y.iloc[va],p),5) for name,p in probabilities.items()},flush=True)
    folds=pd.DataFrame(fold_rows);summary=[]
    for name,g in folds.groupby('model',sort=False):
        summary.append(dict(model=name,**{f'{metric}_{stat}':float(getattr(np,stat)(g[metric]))
            for metric in ['roc_auc','average_precision','accuracy','precision','recall','f1','specificity','npv','log_loss','brier_score'] for stat in ['mean','std']}))
    summary=pd.DataFrame(summary)
    summary.to_csv(OUT/'outer_cv_summary.csv',index=False)
    pd.DataFrame({'actual':y,'outer_fold':outer_fold,**outer_predictions}).to_csv(OUT/'outer_oof_predictions.csv',index=False)
    # Selection is based on mean outer-fold AUC; ties prefer stronger regularization.
    C_scores=[(C,float(summary.loc[summary.model.eq(f'Stacking_C={C:g}'),'roc_auc_mean'].iloc[0])) for C in CS]
    best_C=max(C_scores,key=lambda pair:(pair[1],-pair[0]))[0]
    print('Selected combiner C:',best_C,'; fitting final training-only OOF ensemble',flush=True)
    base_oof,base_val,fitted,audits=fit_members(prototypes,X,y,val[FEATURES])
    pd.DataFrame(audits).to_csv(OUT/'final_inner_fold_audit.csv',index=False)
    meta=combiner(best_C).fit(base_oof,y)
    soft=FittedProbabilityEnsemble(fitted)
    stack=FittedProbabilityEnsemble(fitted,meta)
    joblib.dump(soft,MODELS/'soft_voting.joblib');joblib.dump(stack,MODELS/'stacking.joblib')
    base_table=pd.DataFrame(base_oof,columns=[name for name,_ in fitted])
    base_table.assign(actual=y).to_csv(OUT/'base_training_oof_predictions.csv',index=False)
    base_table.corr().to_csv(OUT/'base_oof_probability_correlations.csv')
    errors=base_table.ge(.5).ne(y.to_numpy(),axis=0)
    disagreement=pd.DataFrame(index=base_table.columns,columns=base_table.columns,dtype=float)
    for a in base_table:
        for b in base_table: disagreement.loc[a,b]=np.mean((base_table[a]>=.5)!=(base_table[b]>=.5))
    disagreement.to_csv(OUT/'base_oof_class_disagreement.csv')
    errors.astype(int).corr().to_csv(OUT/'base_oof_error_correlations.csv')
    probabilities={'XGBoost':base_val[:,0],'SoftVoting':base_val.mean(axis=1),'Stacking':meta.predict_proba(base_val)[:,1]}
    validation=pd.DataFrame([dict(model=name,**evaluate_probabilities(val.cardio,p)) for name,p in probabilities.items()])
    validation.to_csv(OUT/'validation_results.csv',index=False)
    previous=pd.read_csv('outputs/additional_methods/all_methods_validation.csv')
    pd.concat([previous,validation.loc[validation.model.ne('XGBoost')]],ignore_index=True).sort_values('roc_auc',ascending=False,kind='stable').to_csv(OUT/'all_methods_validation.csv',index=False)
    for name,p in probabilities.items():
        pd.DataFrame({'actual':val.cardio,'probability':p,'prediction':(p>=.5).astype(int)}).to_csv(OUT/f'validation_predictions_{name}.csv',index=False)
    np.testing.assert_allclose(probabilities['XGBoost'],joblib.load('models/XGBoost.joblib').predict_proba(val[FEATURES])[:,1],rtol=1e-6,atol=1e-7)
    # Freeze ensemble/model selection before any primary test file is accessed.
    winner=validation.sort_values('roc_auc',ascending=False,kind='stable').iloc[0].model
    selection={'winner':winner,'selected_combiner_C':best_C,'rule':'highest validation ROC-AUC among XGBoost, SoftVoting and Stacking',
        'test_used_for_selection':False,'classification_threshold':0.5,
        'data_reuse_limitations':'Previously inspected source test data and repeatedly reused validation; not independent confirmation'}
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2))
    selected={'XGBoost':fitted[0][1],'SoftVoting':soft,'Stacking':stack}[winner]
    joblib.dump(selected,MODELS/'selected_model.joblib')
    if winner=='XGBoost':
        pred=pd.read_csv('outputs/test_predictions.csv')
        test_metrics=evaluate_probabilities(pred.actual,pred.probability)
        source='Reused saved original test predictions; no new test inference'
    else:
        test=pd.read_csv('data/primary_test.csv')
        assert not profiles(train)&profiles(test) and not profiles(val)&profiles(test)
        with threadpool_limits(limits=1): p=selected.predict_proba(test[FEATURES])[:,1]
        pred=pd.DataFrame({'actual':test.cardio,'probability':p,'prediction':(p>=.5).astype(int)})
        test_metrics=evaluate_probabilities(test.cardio,p)
        source='Exploratory evaluation of validation-selected ensemble on previously inspected test partition'
    pred.to_csv(OUT/'selected_test_predictions.csv',index=False)
    pd.DataFrame([dict(model=winner,**test_metrics,source=source)]).to_csv(OUT/'selected_test_results.csv',index=False)
    make_plots(validation,probabilities,val.cardio)
    assert all(digest(path)==h for path,h in before.items())
    (OUT/'verification.json').write_text(json.dumps({'passed':True,'protected_file_sha256':before,
        'checks':['Outer-fold profile disjointness','Each inner OOF record predicted exactly once without fitting on it',
                  'Every inner/full-training base imputer and scaler verified against its actual fitting records',
                  'Combiner fitted exclusively on training OOF probabilities','Original validation XGBoost probabilities reproduced',
                  'Selection frozen before test access','Original artifacts unchanged']},indent=2))
    coefficients=pd.DataFrame({'member':base_table.columns,'coefficient':meta.coef_[0]})
    coefficients['intercept']=meta.intercept_[0]
    coefficients.to_csv(OUT/'stacking_coefficients.csv',index=False)
    notes=f'''# Heterogeneous ensembles

Compared equal-weight SoftVoting and Stacking (XGBoost + RandomForest + KNN base pipelines, LogisticRegression combiner) against individual XGBoost on the original 70/15/15 split. Seed 42; 14 predictors. No class weighting, resampling, external dataset retraining, threshold selection or post-hoc calibration is added.

Base settings: XGBoost depth 3, 100 trees, learning rate 0.1; RF depth 10, 300 trees; KNN k=101 with uniform Euclidean voting. Original tree pipelines retain median imputation; KNN retains numeric median imputation/scaling and categorical most-frequent imputation/one-hot encoding. Every fitting subset has its own preprocessing.

Five outer training folds evaluate the complete stacking procedure. Within each outer training subset, five inner folds generate three out-of-fold probabilities per record. LogisticRegression combiners C={CS} fit these training OOF predictions and are evaluated on the outer held-out records, whose data never fit either layer or preprocessing. SoftVoting averages the same outer refitted members. Combiner C selected by mean outer-fold ROC-AUC: {best_C}. These scores are tuning estimates (C selected on them); base settings were also previously tuned using the full primary training partition, so this is not fully nested hyperparameter selection. Fold SD is not a confidence interval.

Final fit generates new five-fold OOF probabilities from all 47,968 training records to fit the selected combiner, and refits each member on all training records. No original inputs are passed directly to the combiner. LogisticRegression has intercept and L2 regularization; coefficients need not sum to one or be positive. They are not feature-level clinical effects.

Validation winner: {winner}. {source}. Original XGBoost, other individual models, reports and prior results remain unchanged. Validation and test records were previously examined; these results are exploratory and are not independent confirmation. Subject independence cannot be verified from available IDs.

Validation ROC-AUC, recall, specificity, precision, F1, average precision, NPV, log loss and Brier score are saved. Calibration curves are diagnostics, not proof of calibration. Base OOF correlations and class disagreement quantify similarity; algorithmic diversity alone does not guarantee complementary errors. Ensemble improvements are observed differences, not claims of statistical superiority. The primary XGBoost SHAP plots explain only XGBoost, not the new ensemble.

Scripts: 08_train_ensembles.py; ensemble_models.py; verify_ensembles.py. Reproduce with venv/bin/python 08_train_ensembles.py (CVD_JOBS defaults to 2). Fitted inference artifacts are in models/ensembles/. FittedProbabilityEnsemble is an inference-only container; do not use it for prefit stacking CV.

References:
- Wolpert (1992), Stacked generalization, Neural Networks 5(2), 241–259: https://doi.org/10.1016/S0893-6080(05)80023-1
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
'''
    (OUT/'README.md').write_text(notes)
    print('\nVALIDATION RESULTS\n'+validation.to_string(index=False),flush=True)
    print('\nCV SUMMARY\n'+summary[['model','roc_auc_mean','roc_auc_std']].to_string(index=False),flush=True)
    print('\nSELECTED:',winner,'\nTEST:',test_metrics,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
