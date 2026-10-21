"""Evaluate frozen winner and explain a reproducible test sample using training background."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/cvd-matplotlib')
import json
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from pipeline_common import FEATURES, SEED, evaluate

def main():
    model=joblib.load('models/best_model.joblib')
    test=pd.read_csv('data/primary_test.csv')
    train=pd.read_csv('data/primary_train.csv')
    X,y=test[FEATURES],test.cardio
    metrics=evaluate(model,X,y)
    pd.DataFrame([metrics]).to_csv('outputs/test_results.csv',index=False)
    pd.DataFrame({'actual':y,'probability':model.predict_proba(X)[:,1],'prediction':model.predict(X)}).to_csv('outputs/test_predictions.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,5))
    RocCurveDisplay.from_estimator(model,X,y,ax=ax,name='Selected model')
    ax.plot([0,1],[0,1],'k--',label='Chance'); ax.legend(); ax.set_title('Primary held-out test ROC')
    fig.tight_layout(); fig.savefig('outputs/roc_curve.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5))
    ConfusionMatrixDisplay.from_estimator(model,X,y,ax=ax,display_labels=['No CVD','CVD'],colorbar=False)
    ax.set_title('Primary test; threshold = 0.5'); fig.tight_layout(); fig.savefig('outputs/confusion_matrix.png',dpi=180); plt.close(fig)
    # Explain transformed model inputs; imputer/scaler fit only on training data.
    transform=model[:-1]
    background=transform.transform(train[FEATURES].sample(n=100,random_state=SEED))
    sample=X.sample(n=min(500,len(X)),random_state=SEED)
    transformed=transform.transform(sample)
    estimator=model.named_steps['model']
    if estimator.__class__.__name__=='XGBClassifier':
        # Native exact TreeSHAP avoids the installed SHAP/XGBoost interventional incompatibility.
        import xgboost as xgb
        matrix=xgb.DMatrix(transformed,feature_names=estimator.get_booster().feature_names)
        contributions=estimator.get_booster().predict(matrix,pred_contribs=True)
        margin=estimator.get_booster().predict(matrix,output_margin=True)
        np.testing.assert_allclose(contributions.sum(axis=1),margin,rtol=1e-5,atol=1e-5)
        values=shap.Explanation(values=contributions[:,:-1],base_values=contributions[:,-1],data=transformed,feature_names=FEATURES)
    elif estimator.__class__.__name__=='LogisticRegression':
        explainer=shap.LinearExplainer(estimator,background,feature_names=FEATURES)
        values=explainer(transformed)
    else:
        explainer=shap.TreeExplainer(estimator,data=background,feature_names=FEATURES)
        values=explainer(transformed)
    if values.values.ndim==3: values=values[:,:,1]
    shap.plots.beeswarm(values,max_display=len(FEATURES),show=False)
    plt.gcf().set_size_inches(10,7); plt.tight_layout(); plt.savefig('outputs/shap_summary.png',dpi=180); plt.close()
    shap.plots.waterfall(values[0],max_display=len(FEATURES),show=False)
    plt.tight_layout(); plt.savefig('outputs/shap_waterfall_example.png',dpi=180); plt.close()
    pd.DataFrame({'feature':FEATURES,'mean_absolute_shap':np.abs(values.values).mean(axis=0)}).sort_values('mean_absolute_shap',ascending=False).to_csv('outputs/shap_importance.csv',index=False)
    with open('outputs/shap_metadata.json','w') as f:
        json.dump({'test_sample_size':len(sample),'training_background_size':0 if estimator.__class__.__name__=='XGBClassifier' else len(background), 'method':'native exact path-dependent TreeSHAP using training tree cover' if estimator.__class__.__name__=='XGBClassifier' else 'SHAP with training background','random_state':SEED,'sample_row_indices':sample.index.tolist(),'output_units':'raw margin (log odds) for XGBoost/LogisticRegression; probability for RandomForest','feature_values':'scaled units for LogisticRegression; original units for trees'},f,indent=2)
    print(metrics,flush=True)

if __name__=='__main__': main()
