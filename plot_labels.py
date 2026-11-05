"""Consistent numeric chart labels; use source values, never edited image pixels."""
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

def label_bars(ax,bars,values=None,fmt='.4f',padding=4,**kwargs):
    values=[b.get_height() for b in bars] if values is None else values
    labels=[format(v,fmt) if v else '' for v in values]
    return ax.bar_label(bars,labels=labels,padding=padding,**kwargs)

def label_cv(ax,means,stds):
    for i,(mean,std) in enumerate(zip(means,stds)):
        ax.annotate(f'{mean:.4f} ± {std:.4f}',(i,mean),xytext=(0,-16),
                    textcoords='offset points',ha='center',va='top',fontsize=8,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2))

def draw_roc(ax,y,score,name):
    fpr,tpr,_=roc_curve(y,score)
    ax.plot(fpr,tpr,label=f'{name} (AUC = {roc_auc_score(y,score):.4f})')
    ax.set(xlabel='False positive rate',ylabel='True positive rate (recall)')
    ax.legend(loc='lower right',fontsize=9)

def draw_pr(ax,y,score,name):
    precision,recall,_=precision_recall_curve(y,score)
    ax.plot(recall,precision,label=f'{name} (AP = {average_precision_score(y,score):.4f})')
    ax.set(xlabel='Recall',ylabel='Precision')
    ax.legend(loc='lower left',fontsize=9)

def label_shap_beeswarm(ax,values):
    names=list(values.feature_names)
    importance=dict(zip(names,np.abs(values.values).mean(axis=0)))
    labels=[f'{t.get_text()}  |  {importance[t.get_text()]:.3f}'
            if t.get_text() in importance else t.get_text() for t in ax.get_yticklabels()]
    ax.set_yticks(ax.get_yticks(),labels=labels)
    ax.set_xlabel('SHAP contribution (log-odds)')
    ax.set_title('Feature labels include mean absolute SHAP (log-odds)',fontsize=11,pad=15)
