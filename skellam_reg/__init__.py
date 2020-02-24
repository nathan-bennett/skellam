#!/usr/bin/env python
import numpy as np
from scipy.optimize import minimize
from metrics import SkellamMetrics
import warnings
from scipy.stats import skellam
from shared_utils import ArrayUtils


class SkellamRegression:
    def __init__(self, x, y, l0, l1, add_intercept=True):
        self.y = y
        self.l0 = l0
        self.l1 = l1
        self.add_intercept = add_intercept
        self.coeff_size = None
        self.x0, self.x1 = self.split_or_duplicate_x(x, self.add_intercept)

    @staticmethod
    def split_or_duplicate_x(x, add_intercept):
        return ArrayUtils.split_or_duplicate_x(x, add_intercept)

    @staticmethod
    def _skellam_pmf(x, mu0, mu1):
        """
        This is the probability mass function of the skellam distribution taken directly from the scipy stats package.
        """
        px = skellam.pmf(x, mu1=mu0, mu2=mu1, loc=0)
        return px

    def _log_likelihood(self, coefficients):
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
                self._log_likelihood,
                x0,
                method=optimization_method,
                options={"disp": display_optimisation},
            )

            x0 = first_run.x
        else:
            if x0.shape[0] != x_shape:
                raise ValueError("Initial numbers are not equal to: {}".format(x_shape))

        results = minimize(
            self._log_likelihood,
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

    def predict(self, _x, add_intercept=True):
        """Using the model created previously, this will predict values of y based on a new array x
        """
        # convert x to be in the correct format - a 2 dimensional numpy array
        _x0, _x1 = self.split_or_duplicate_x(_x, add_intercept)

        lambda_0_coefficients = self._results.x[0 : self.coeff_size].reshape(-1, 1)
        lambda_1_coefficients = self._results.x[self.coeff_size :].reshape(-1, 1)

        _lambda0 = np.squeeze(_x0 @ lambda_0_coefficients)
        _lambda1 = np.squeeze(_x1 @ lambda_1_coefficients)

        y_hat = np.exp(_lambda0) - np.exp(_lambda1)

        return y_hat

    def model_performance(self, test_x=None, test_y=None):
        """Calculate key metrics such as r2
        """
        train_x_values = [self.x0, self.x1]
        if test_x is not None and test_y is not None:
            test_x_0, test_x_1 = self.split_or_duplicate_x(test_x, self.add_intercept)
            test_x_values = [test_x_0, test_x_1]
            # We do not add intercept as it already has been added
            predictions = self.predict(test_x_values, False)
            return SkellamMetrics(
                test_x_values,
                test_y,
                predictions,
                self._results,
                self.l0,
                self.l1,
                train_x_values,
            )
        else:
            # We do not add intercept as it already has been added
            predictions = self.predict(train_x_values, False)
            return SkellamMetrics(
                train_x_values,
                self.y,
                predictions,
                self._results,
                self.l0,
                self.l1,
                train_x_values,
            )
