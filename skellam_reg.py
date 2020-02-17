#!/usr/bin/env python
import numpy as np
from scipy.optimize import minimize
from metrics.skellam_metrics import SkellamMetrics
import pandas as pd
from scipy.special import ive, xlogy
import warnings


class SkellamRegression:

    def __init__(self, x, y, l1, l2, add_coefficients=True):
        self.y = y
        self.l1 = l1
        self.l2 = l2
        self.add_coefficients = add_coefficients
        self.coeff_size = None
        # convert x to be in the correct format - a 2 dimensional numpy array
        self.x = self.convert_to_array(x)

    @staticmethod
    def convert_to_array(_x):
        if isinstance(_x, np.ndarray) and _x.ndim == 1:
            return _x.reshape(-1, 1)
        elif isinstance(_x, pd.core.series.Series):
            return _x.values.reshape(-1, 1)
        else:
            return _x

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
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # initial estimate
        if x0 is None:
            x0 = np.ones(self.x.shape[1] * 2)

            first_run = minimize(self.log_likelihood,
                                 x0,
                                 method=optimization_method,
                                 options={'disp': display_optimisation})

            x0 = first_run.x
        else:
            if x0.shape[0] != self.x.shape[1] * 2:
                raise ValueError

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

    def predict(self, _x):
        """Using the model created previously, this will predict values of y based on a new array x
        """
        _x = self.convert_to_array(_x)

        lambda_1_coefficients = self._results.x[0: self.coeff_size].reshape(-1, 1)
        lambda_2_coefficients = self._results.x[self.coeff_size:].reshape(-1, 1)

        _lambda1 = np.squeeze(_x @ lambda_1_coefficients)
        _lambda2 = np.squeeze(_x @ lambda_2_coefficients)

        y_hat = np.exp(_lambda1) - np.exp(_lambda2)

        return y_hat

    def model_performance(self, test_x=None, test_y=None):
        """Calculate key metrics such as r2
        """
        if test_x is not None and test_y is not None:
            test_x = self.convert_to_array(test_x)
            predictions = self.predict(test_x)
            return SkellamMetrics(test_x, test_y, predictions, self._results, self.l1, self.l2)
        else:
            predictions = self.predict(self.x)
            return SkellamMetrics(self.x, self.y, predictions, self._results, self.l1, self.l2)

