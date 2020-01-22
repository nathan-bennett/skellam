#!/usr/bin/env python
from scipy.stats import skellam
import numpy as np
from scipy.optimize import minimize
from metrics.skellam_metrics import SkellamMetrics
import pandas as pd


class SkellamRegression:

    def __init__(self, x, y):
        self.y = y
        # convert x to be in the correct format - a 2 dimensional numpy array
        if isinstance(x, np.ndarray) and x.ndim == 1:
            self.x = x.reshape(-1, 1)
        elif isinstance(x, pd.core.series.Series):
            self.x = x.values.reshape(-1, 1)
        else:
            self.x = x

    def log_likelihood(self, coefficients):

        self.coeff_size = len(coefficients) // 2
        coefficients1 = coefficients[0:self.coeff_size].reshape(-1, 1)
        coefficients2 = coefficients[self.coeff_size:].reshape(-1, 1)

        lambda1 = np.squeeze(self.x @ coefficients1)
        lambda2 = np.squeeze(self.x @ coefficients2)

        neg_log_likelihood = -np.sum(skellam.logpmf(self.y, mu1=np.exp(lambda1), mu2=np.exp(lambda2), loc=0))

        return neg_log_likelihood

    def _train(self, x0, optimization_method, display_optimisation):
        # initial estimate
        if x0 is None:
            x0 = np.ones(self.x.shape[1] * 2)
        else:
            if x0.shape[0] != self.x.shape[1] * 2:
                raise ValueError

        results = minimize(self.log_likelihood,
                           x0,
                           method=optimization_method,
                           options={'disp': display_optimisation})

        self._results = results

        return results

    def train(self, x0=None, optimization_method="SLSQP", display_optimisation=True):
        return self._train(x0, optimization_method, display_optimisation)

    def predict(self, x):
        if isinstance(x, np.ndarray) and x.ndim == 1:
            x = x.reshape(-1, 1)
        elif isinstance(x, pd.core.series.Series):
            x = x.values.reshape(-1, 1)

        lambda_1_coefficients = self._results.x[0: self.coeff_size].reshape(-1, 1)
        lambda_2_coefficients = self._results.x[self.coeff_size:].reshape(-1, 1)

        _lambda1 = np.exp(np.squeeze(x @ lambda_1_coefficients))
        _lambda2 = np.exp(np.squeeze(x @ lambda_2_coefficients))

        y_hat = _lambda1 - _lambda2

        return y_hat

    def model_performance(self):
        predictions = self.predict(self.x)
        return SkellamMetrics(self.x, self.y, predictions)
