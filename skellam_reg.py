#!/usr/bin/env python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import skellam
from metrics.skellam_metrics import SkellamMetrics
import pandas as pd
import warnings


class SkellamRegression:

    def __init__(self, x, y, add_coefficients=True):
        self.y = y
        self.add_coefficients = add_coefficients
        self.coeff_size = None
        self.x = self._convert_to_array(x)

    @staticmethod
    def _convert_to_array(_x):
        """ Convert x to be in the correct format - a 2 dimensional numpy array
        """
        if isinstance(_x, np.ndarray) and _x.ndim == 1:
            return _x.reshape(-1, 1)
        elif isinstance(_x, pd.core.series.Series):
            return _x.values.reshape(-1, 1)
        else:
            return _x

    def log_likelihood(self, coefficients):
        """Function to calculate the negative log likelihood of the Skellam distribution
        """
        self.coeff_size = len(coefficients) // 2
        coefficients1 = coefficients[0:self.coeff_size].reshape(-1, 1)
        coefficients2 = coefficients[self.coeff_size:].reshape(-1, 1)

        lambda1 = np.squeeze(self.x @ coefficients1)
        lambda2 = np.squeeze(self.x @ coefficients2)

        neg_ll = -np.sum(skellam.logpmf(self.y, np.exp(lambda1), np.exp(lambda2)))

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
        x = self._convert_to_array(x)

        lambda_1_coefficients = self._results.x[0: self.coeff_size].reshape(-1, 1)
        lambda_2_coefficients = self._results.x[self.coeff_size:].reshape(-1, 1)

        _lambda1 = np.squeeze(x @ lambda_1_coefficients)
        _lambda2 = np.squeeze(x @ lambda_2_coefficients)

        y_hat = np.exp(_lambda1) - np.exp(_lambda2)

        return y_hat

    def model_performance(self, test_x=None, test_y=None):
        """Calculate key metrics such as r2
        """
        if test_x is not None and test_y is not None:
            test_x = self._convert_to_array(test_x)
            predictions = self.predict(test_x)
            return SkellamMetrics(test_x, test_y, predictions)
        else:
            predictions = self.predict(self.x)
            return SkellamMetrics(self.x, self.y, predictions)

