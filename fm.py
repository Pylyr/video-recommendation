import argparse
from datetime import datetime
from pathlib import Path
import json
import os
from sklearn.metrics import roc_auc_score, log_loss
from scipy.sparse import load_npz
import pywFM
import numpy as np
import sklearn
import scipy


# Location of libFM's compiled binary file
os.environ['LIBFM_PATH'] = str(Path('libfm/bin').absolute()) + '/'


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class FMClassifier(sklearn.base.BaseEstimator):
    def __init__(self, embedding_size=20, nb_iterations=40):
        super().__init__()
        self.embedding_size = embedding_size
        self.nb_iterations = nb_iterations

    def fit(self, X, y):
        """
        X is usually sparse, nb_samples x nb_features
        y is binary
        """
        fm = pywFM.FM(task='classification', num_iter=self.nb_iterations,
                      k2=self.embedding_size, rlog=True)  # MCMC method
        # rlog contains the RMSE at each epoch, we do not need it here
        model = fm.run(X, y, X, y)

        # Store parameters
        self.mu = model.global_bias
        self.W = np.array(model.weights)
        self.V = model.pairwise_interactions
        self.V2 = np.power(self.V, 2)
        self.rlog = model.rlog
        return self

    def predict_proba(self, X):
        X2 = X.copy()
        if scipy.sparse.issparse(X):
            X2.data **= 2
        else:
            X2 **= 2

        y_pred = (self.mu + X @ self.W +
                  0.5 * (np.power(X @ self.V, 2).sum(axis=1)
                         - (X2 @ self.V2).sum(axis=1)).A1)
        return sigmoid(y_pred)
