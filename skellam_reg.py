#!/usr/bin/env python
import numpy as np
from scipy.optimize import minimize
from metrics.skellam_metrics import SkellamMetrics
import pandas as pd
from scipy.special import ive, xlogy
import warnings


class SkellamRegression:

    def __init__(self, x, y):
        self.y = y
        self.coeff_size = None
        # convert x to be in the correct format - a 2 dimensional numpy array
        if isinstance(x, np.ndarray) and x.ndim == 1:
            self.x = x.reshape(-1, 1)
        elif isinstance(x, pd.core.series.Series):
            self.x = x.values.reshape(-1, 1)
        else:
            self.x = x

    def _non_central_x2_pmf(self, x, df, nc):
        """This is the probability mass function of the non-central chi-squared distribution
        This was derived from the scipy stats package.
        """
        df2 = df / 2.0 - 1.0
        xs, ns = np.sqrt(x), np.sqrt(nc)
        res = xlogy(df2 / 2.0, x / nc) - 0.5 * (xs - ns) ** 2 + np.log(ive(df2, xs * ns) / 2.0)
        return res

    def _skellam_pmf(self, x, mu1, mu2):
        """
        This is the probability mass function of the skellam distribution
        This was derived from the scipy stats package.
        """
        px = np.where(x < 0,
                      np.exp(self._non_central_x2_pmf(2 * mu2, 2 * (1 - x), 2 * mu1) * 2),
                      np.exp(self._non_central_x2_pmf(2 * mu1, 2 * (1 + x), 2 * mu2) * 2))
        return px

    def log_likelihood(self, coefficients):
        """Function to calculate the negative log likelihood of the skellam distribution
        """
        self.coeff_size = len(coefficients) // 2
        coefficients1 = coefficients[0:self.coeff_size].reshape(-1, 1)
        coefficients2 = coefficients[self.coeff_size:].reshape(-1, 1)

        lambda1 = np.squeeze(self.x @ coefficients1)
        lambda2 = np.squeeze(self.x @ coefficients2)

        neg_ll = -np.sum(np.log(self._skellam_pmf(self.y, np.exp(lambda1), np.exp(lambda2))))

        return neg_ll

    def _train(self, x0, optimization_method, display_optimisation):
        # initial estimate
        if x0 is None:
            x0 = np.ones(self.x.shape[1] * 2)
        else:
            if x0.shape[0] != self.x.shape[1] * 2:
                raise ValueError

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        results = minimize(self.log_likelihood,
                           x0,
                           method=optimization_method,
                           options={'disp': display_optimisation})

        self._results = results

        return results

    def train(self, x0=None, optimization_method="SLSQP", display_optimisation=False):
        """Minmizes the negative log likelihood to find the optimal values for our coefficients
        """
        return self._train(x0, optimization_method, display_optimisation)

    def predict(self, x):
        """Using the model created previously, this will predict values of y based on a new array x
        """
        if isinstance(x, np.ndarray) and x.ndim == 1:
            x = x.reshape(-1, 1)
        elif isinstance(x, pd.core.series.Series):
            x = x.values.reshape(-1, 1)

        lambda_1_coefficients = self._results.x[0: self.coeff_size].reshape(-1, 1)
        lambda_2_coefficients = self._results.x[self.coeff_size:].reshape(-1, 1)

        _lambda1 = np.squeeze(x @ lambda_1_coefficients)
        _lambda2 = np.squeeze(x @ lambda_2_coefficients)

        y_hat = np.exp(_lambda1) - np.exp(_lambda2)

        return y_hat

    def model_performance(self):
        """Calculate key metrics such as r2
        """
        predictions = self.predict(self.x)
        return SkellamMetrics(self.x, self.y, predictions)
