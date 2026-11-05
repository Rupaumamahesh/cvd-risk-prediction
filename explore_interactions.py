"""Numeric native-XGBoost TreeSHAP interaction summary; no model fitting."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/cvd-matplotlib')
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    model=joblib.load('models/best_model.joblib')
    features=joblib.load('models/feature_list.joblib')
    sample=pd.read_csv('data/primary_test.csv')[features].sample(200,random_state=42)
    estimator=model.named_steps['model']
    if estimator.__class__.__name__!='XGBClassifier':
        raise ValueError('This interaction plot requires the selected XGBoost model.')
    matrix=xgb.DMatrix(model[:-1].transform(sample),feature_names=estimator.get_booster().feature_names)
    interactions=estimator.get_booster().predict(matrix,pred_interactions=True)
    margin=estimator.get_booster().predict(matrix,output_margin=True)
    np.testing.assert_allclose(interactions.sum(axis=(1,2)),margin,atol=1e-5,rtol=1e-5)
    mean=np.abs(interactions[:,:-1,:-1]).mean(axis=0)
    pd.DataFrame(mean,index=features,columns=features).to_csv('outputs/shap_interaction_mean_absolute.csv')
    fig,ax=plt.subplots(figsize=(12,10))
    im=ax.imshow(mean,cmap='Blues')
    ax.set_xticks(range(len(features)),features,rotation=45,ha='right')
    ax.set_yticks(range(len(features)),features)
    for i in range(len(features)):
        for j in range(len(features)):
            ax.text(j,i,f'{mean[i,j]:.4f}',ha='center',va='center',fontsize=7,
                    color='white' if mean[i,j]>mean.max()*.55 else 'black')
    fig.colorbar(im,ax=ax,label='Mean absolute TreeSHAP interaction contribution (log-odds)')
    ax.set_title('XGBoost SHAP interactions — 200 test records; seed 42\nDiagonal: main effects; off-diagonal: allocated pair interactions',fontsize=12)
    fig.tight_layout();fig.savefig('outputs/shap_interaction_summary.png',dpi=220);plt.close(fig)
    print('Saved numeric SHAP interaction chart; additive reconstruction verified.')

if __name__=='__main__': main()
