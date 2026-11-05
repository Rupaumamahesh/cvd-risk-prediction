"""Serializable probability ensembles and training-preprocessing assertions."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class FittedProbabilityEnsemble(ClassifierMixin, BaseEstimator):
    """Inference container for full-training fitted members and an OOF-trained combiner.

    This object deliberately has no fit method: 08_train_ensembles.py performs
    and audits the training procedure explicitly. Never use this prefit object
    as a substitute for training the complete procedure within a CV fold.
    """
    def __init__(self, estimators, combiner=None):
        self.estimators = estimators
        self.combiner = combiner
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = estimators[0][1].n_features_in_
        self.feature_names_in_ = estimators[0][1].feature_names_in_

    def member_probabilities(self, X):
        return np.column_stack([model.predict_proba(X)[:,1] for _,model in self.estimators])

    def predict_proba(self, X):
        base = self.member_probabilities(X)
        p = base.mean(axis=1) if self.combiner is None else self.combiner.predict_proba(base)[:,1]
        return np.column_stack([1-p,p])

    def predict(self, X):
        return (self.predict_proba(X)[:,1] >= 0.5).astype(int)


def assert_training_preprocessing(model, X):
    """Verify medians and numeric means against the actual fitting records."""
    if 'imputer' in model.named_steps:
        np.testing.assert_allclose(model['imputer'].statistics_,X.median())
        if model['scale'] != 'passthrough':
            np.testing.assert_allclose(model['scale'].mean_,model['imputer'].transform(X).mean(axis=0))
    else:
        pre = model['preprocess']
        numeric = pre.named_transformers_['numeric']
        columns = pre.transformers_[0][2]
        np.testing.assert_allclose(numeric['imputer'].statistics_,X[columns].median())
        np.testing.assert_allclose(numeric['scale'].mean_,numeric['imputer'].transform(X[columns]).mean(axis=0))
        cat = pre.named_transformers_['categorical']
        for column, value in zip(pre.transformers_[1][2],cat['imputer'].statistics_):
            assert value in X[column].mode().values
