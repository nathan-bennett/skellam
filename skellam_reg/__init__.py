#!/usr/bin/env python
import numpy as np
from scipy.optimize import minimize
from skellam_reg.metrics import SkellamMetrics
import pandas as pd
import warnings
from scipy.stats import skellam


class SkellamRegression:
    def __init__(self, x, y, l1, l2, add_coefficients=True):
        self.y = y
        self.l1 = l1
        self.l2 = l2
        self.add_coefficients = add_coefficients
        self.coeff_size = None
        self.x0, self.x1 = self.split_or_duplicate_x(x)

    def convert_to_array(self, _x):
        if isinstance(_x, np.ndarray) and _x.ndim == 1:
            return _x.reshape(-1, 1)
        elif isinstance(_x, pd.core.series.Series):
            return _x.values.reshape(-1, 1)
        else:
            return _x

    def split_or_duplicate_x(self, x):
        """This function aims to to create x0 and x1 by either duplicating x, if x is an array or series, otherwise
        if x is a list then we will split the list where the first element will be equal to x0 whilst the second
        element will be equal to x1
        """
        x0 = None
        x1 = None
        if (
            isinstance(x, np.ndarray)
            or isinstance(x, pd.core.series.Series)
            or isinstance(x, pd.core.frame.DataFrame)
        ):
            x0, x1 = x, x
        elif len(x) == 2:
            if (
                isinstance(x[0], np.ndarray)
                or isinstance(x[0], pd.core.series.Series)
                or isinstance(x, pd.core.frame.DataFrame)
            ):
                x0 = self.convert_to_array(x[0])
            if (
                isinstance(x[1], np.ndarray)
                or isinstance(x[1], pd.core.series.Series)
                or isinstance(x, pd.core.frame.DataFrame)
            ):
                x1 = self.convert_to_array(x[1])
            else:
                x0 = x[0]
                x1 = x[1]
        if x0 is None and x1 is None:
            raise ValueError(
                "x must either be an a list of two arrays or a single array"
            )
        return x0, x1

    def _skellam_pmf(self, x, mu0, mu1):
        """
        This is the probability mass function of the skellam distribution taken directly from the scipy stats package.
        """
        px = skellam.pmf(x, mu1=mu0, mu2=mu1, loc=0)
        return px

    def log_likelihood(self, coefficients):
        """Function to calculate the negative log likelihood of the skellam distribution
        """
        self.coeff_size = self.x0.shape[1]
        coefficients1 = coefficients[0 : self.coeff_size].reshape(-1, 1)
        coefficients2 = coefficients[self.coeff_size :].reshape(-1, 1)

        lambda0 = np.squeeze(self.x0 @ coefficients1)
        lambda1 = np.squeeze(self.x1 @ coefficients2)

        neg_ll = -np.sum(
            np.log(self._skellam_pmf(self.y, np.exp(lambda0), np.exp(lambda1)))
        )

        return neg_ll

    def _train(self, x0, optimization_method, display_optimisation):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        x_shape = self.x0.shape[1] + self.x1.shape[1]
        # initial estimate
        if x0 is None:
            x0 = np.ones(x_shape)

            first_run = minimize(
                self.log_likelihood,
                x0,
                method=optimization_method,
                options={"disp": display_optimisation},
            )

            x0 = first_run.x
        else:
            if x0.shape[0] != x_shape:
                raise ValueError("Initial numbers are not equal to: {}".format(x_shape))

        results = minimize(
            self.log_likelihood,
            x0,
            method=optimization_method,
            options={"disp": display_optimisation},
        )

        self._results = results

        return results

    def train(self, x0=None, optimization_method="SLSQP", display_optimisation=False):
        """Minimizes the negative log likelihood to find the optimal values for our coefficients
        """
        return self._train(x0, optimization_method, display_optimisation)

    def predict(self, _x):
        """Using the model created previously, this will predict values of y based on a new array x
        """
        # convert x to be in the correct format - a 2 dimensional numpy array
        _x0, _x1 = self.split_or_duplicate_x(_x)

        lambda_0_coefficients = self._results.x[0 : self.coeff_size].reshape(-1, 1)
        lambda_1_coefficients = self._results.x[self.coeff_size :].reshape(-1, 1)

        _lambda0 = np.squeeze(_x0 @ lambda_0_coefficients)
        _lambda1 = np.squeeze(_x1 @ lambda_1_coefficients)

        y_hat = np.exp(_lambda0) - np.exp(_lambda1)

        return y_hat

    def model_performance(self, test_x=None, test_y=None):
        """Calculate key metrics such as r2
        """
        if test_x is not None and test_y is not None:
            test_x_0, test_x_1 = self.split_or_duplicate_x(test_x)
            test_x_values = [test_x_0, test_x_1]
            predictions = self.predict(test_x_values)
            return SkellamMetrics(
                test_x_values, test_y, predictions, self._results, self.l1, self.l2
            )
        else:
            train_x_values = [self.x0, self.x1]
            predictions = self.predict(train_x_values)
            return SkellamMetrics(
                train_x_values, self.y, predictions, self._results, self.l1, self.l2
            )
