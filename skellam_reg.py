#!/usr/bin/env python
from scipy.stats import skellam
import numpy as np
from scipy.optimize import minimize


class SkellamRegression:

    def __init__(self, x, y):
        self.y = y
        if type(x) == np.ndarray:
            self.x = np.matrix(x)
        else:
            self.x = x

    def log_likelihood(self, coefficients):
        coefficients1 = coefficients[0:len(coefficients) // 2]
        coefficients2 = coefficients[len(coefficients) // 2:]

        lambda1 = np.squeeze(
            np.asarray(self.x @ np.matrix(coefficients1).T)
        )
        lambda2 = np.squeeze(
            np.asarray(self.x @ np.matrix(coefficients2).T)
        )

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
        if type(x) == np.ndarray:
            x = np.matrix(x)

        lambda_1_coefficients = self._results.x[0:len(self._results.x) // 2]
        lambda_2_coefficients = self._results.x[len(self._results.x) // 2:]

        _lambda1 = np.exp(
            np.squeeze(
                np.asarray(x @ np.matrix(lambda_1_coefficients).T)
            )
        )
        _lambda2 = np.exp(
            np.squeeze(
                np.asarray(x @ np.matrix(lambda_2_coefficients).T)
            )
        )

        y_hat = _lambda1 - _lambda2

        return y_hat
