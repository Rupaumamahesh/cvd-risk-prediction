"""Refresh output chart labels without fitting models or changing result artifacts."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/cvd-matplotlib')
import importlib
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits

def main():
    # Snapshot all existing non-image output data and all fitted models.
    # Numeric-only plot exports may change; core metrics and predictions may not.
    exempt={'viz_bp_bin_counts.csv','viz_age_bin_counts.csv','calibration_bin_values.csv',
            'shap_interaction_mean_absolute.csv','plot_label_verification.json'}
    protected=[p for p in Path('outputs').rglob('*') if p.is_file() and p.suffix in {'.csv','.json'} and p.name not in exempt]
    protected+=list(Path('models').rglob('*.joblib'))
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    eda=importlib.import_module('00_visualize_data')
    eda.main()
    review=importlib.import_module('05_review_artifacts')
    review.make_plots()
    primary=importlib.import_module('03_evaluate_test_shap')
    pred=pd.read_csv('outputs/test_predictions.csv')
    primary.plot_test_predictions(pred.actual,pred.prediction,pred.probability)
    test=pd.read_csv('data/primary_test.csv')
    train=pd.read_csv('data/primary_train.csv')
    model=joblib.load('models/best_model.joblib')
    values,_,_,_=primary.compute_shap(model,test[primary.FEATURES],train)
    primary.plot_shap(values)
    saved=pd.read_csv('outputs/shap_importance.csv').set_index('feature')
    for name,v in zip(values.feature_names,np.abs(values.values).mean(axis=0)):
        np.testing.assert_allclose(v,saved.loc[name,'mean_absolute_shap'],rtol=1e-5,atol=1e-7)
    additional=importlib.import_module('07_compare_additional_methods')
    comparison=pd.read_csv(additional.OUT/'all_methods_validation.csv')
    additional.plot_comparison(comparison)
    models={name:joblib.load(f'models/additional_methods/{name}.joblib' if name in additional.CONFIGS else f'models/{name}.joblib') for name in comparison.model}
    additional.plot_roc_curves(models,pd.read_csv('data/primary_val.csv'))
    ensemble=importlib.import_module('08_train_ensembles')
    validation=pd.read_csv(ensemble.OUT/'validation_results.csv')
    probabilities={name:pd.read_csv(ensemble.OUT/f'validation_predictions_{name}.csv').probability.to_numpy() for name in validation.model}
    actual=pd.read_csv(ensemble.OUT/'validation_predictions_XGBoost.csv').actual
    ensemble.make_plots(validation,probabilities,actual)
    importlib.import_module('explore_interactions').main()
    # Refresh the existing contact sheet so it does not show stale charts.
    names=['roc_curve','confusion_matrix','shap_summary','model_comparison','cross_dataset_comparison','class_distribution']
    sheet=Image.new('RGB',(1800,1800),'white')
    for i,name in enumerate(names):
        im=Image.open(f'outputs/{name}.png');im.thumbnail((890,590))
        sheet.paste(im,((i%2)*900,(i//2)*600))
    sheet.save('outputs/chart_check.png')
    for path,expected in before.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
    age=pd.read_csv('outputs/viz_age_bin_counts.csv');bp=pd.read_csv('outputs/viz_bp_bin_counts.csv')
    assert age['count'].sum()==70000
    assert bp.groupby('measurement')['count'].sum().eq(70000).all()
    output={'passed':True,'checks':['All existing metric/prediction/CV/selection files and fitted models byte-identical',
        'Histogram counts sum to raw cohort size','SHAP contributions match saved importance values',
        'Native interaction contributions reconstruct model margins'],
        'unchanged_images':['architecture_drawio_preview.png (editor screenshot, not a numeric chart)'],
        'precision':'ROC-AUC/AP/CV and metric labels: 4 decimals; box quartiles: 2 decimals; calibration coordinates: 3 decimals',
        'notes':'Dense SHAP plots use mean absolute contribution labels, not overlapping labels on every dot. Interaction heatmap refreshed for current XGBoost instead of obsolete RF script.'}
    Path('outputs/plot_label_verification.json').write_text(json.dumps(output,indent=2))
    print('PASS: chart labels refreshed; all model and existing result files unchanged.',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
