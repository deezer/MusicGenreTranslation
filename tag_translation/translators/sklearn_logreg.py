"""
    Translator based on any multilabel sklearn classifier
"""
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from scipy.special import expit as sigmoid

from .translator import LogRegTranslator
from ..conf import N_JOBS, SOLVER


class LogRegMultiOutputClassifier(MultiOutputClassifier):

    def get_estimator_parameters(self):
        coef_mat = np.vstack([estimator.coef_.flatten() for estimator in self.estimators_]).T
        intercept_vec = np.array([estimator.intercept_[0] for estimator in self.estimators_])
        return coef_mat, intercept_vec

    def predict_proba(self, X):
        coef_mat, intercept_vec = self.get_estimator_parameters()
        return sigmoid(X.dot(coef_mat) + intercept_vec)

    def fit(self, X, y, sample_weight=None):

        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement a fit method")

        self.estimators_ = []
        c = 0
        for i in range(y.shape[1]):
            # print("Fitting estimator for label {}".format(self.mlb_target.classes_[i]))
            if np.sum(y[:, i]) > 0:  # or (kb_tr_table is None and np.sum(y[:, i]) > 0):
                estimator = sklearn.base.clone(self.estimator)
                estimator.fit(X, y[:, i])
            else:
                c += 1
                estimator = sklearn.base.clone(self.estimator)
                estimator.coef_ = np.zeros((1, X.shape[1]))
                estimator.intercept_ = np.array([-10e7])
            self.estimators_.append(estimator)
        print("{}/{} tags were missing".format(c, y.shape[1]))
        return self


class SKLearnTranslator(LogRegTranslator):

    def train_and_evaluate(self, train_data, target_data, eval_data, eval_target, score_function):
        print("Training SKlearn-based ml logistic regression")
        self.model = sklearn.base.clone(
            LogRegMultiOutputClassifier(LogisticRegression(solver=SOLVER), n_jobs=N_JOBS))
        self.model.fit(train_data, target_data)
        W, b = self.model.get_estimator_parameters()
        self.W = W.T
        self.b = b.reshape((-1, 1))
        self._evaluate(eval_data, eval_target, score_function)
