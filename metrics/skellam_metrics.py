#!/usr/bin/env python
import numpy as np
import scipy
import pandas as pd


class SkellamMetrics:

    def __init__(self, x, y, y_hat, model, l0, l1):
        self._y = y
        self._y_hat = y_hat
        self.model = model
        self.l0 = self.convert_to_array(l0)
        self.l1 = self.convert_to_array(l1)
        self._x0, self._x1 = self._split_or_duplicate_x(x)

        self.coeff_size = self._x0.shape[1]
        self.lambda_0_coefficients = self.model.x[0: self.coeff_size].reshape(-1, 1)
        self.lambda_1_coefficients = self.model.x[self.coeff_size:].reshape(-1, 1)

    @staticmethod
    def convert_to_array(_x):
        if isinstance(_x, np.ndarray) and _x.ndim == 1:
            return _x.reshape(-1, 1)
        elif isinstance(_x, pd.core.series.Series):
            return _x.values.reshape(-1, 1)
        else:
            return _x
        
    def _split_or_duplicate_x(self, x):
        """This function aims to to create x0 and x1 by either duplicating x, if x is an array or series, otherwise
        if x is a list then we will split the list where the first element will be equal to x0 whilst the second
        element will be equal to x1
        """
        if isinstance(x, np.ndarray) or isinstance(x, pd.core.series.Series) or isinstance(x, pd.core.frame.DataFrame):
            x0, x1 = x, x
        elif len(x) == 2:
            if isinstance(x[0], np.ndarray) or isinstance(x[0], pd.core.series.Series) or isinstance(x, pd.core.frame.DataFrame):
                x0 = self.convert_to_array(x[0])
            else:
                x0 = x[0]
            if isinstance(x[1], np.ndarray) or isinstance(x[1], pd.core.series.Series) or isinstance(x, pd.core.frame.DataFrame):
                x1 = self.convert_to_array(x[1])
            else:
                x1 = x[1]
        else:
            raise ValueError("x must either be an a list of two arrays or a single array")
        return x0, x1

    def sse(self):
        return ((self._y - self._y_hat)**2).sum()

    def _y_bar(self):
        return self._y.mean()

    def sst(self):
        return ((self._y - self._y_bar())**2).sum()

    def r2(self):
        """Calculate R2 for either the train model """
        sse_sst = self.sse()/self.sst()
        return 1-sse_sst

    def _calculate_lambda(self):
        """Create arrays for our predictions of the two Poisson distributions
        """
        _lambda0 = self.convert_to_array(np.exp(np.squeeze(self._x0 @ self.lambda_0_coefficients)))
        _lambda1 = self.convert_to_array(np.exp(np.squeeze(self._x1 @ self.lambda_1_coefficients)))
        return _lambda0, _lambda1

    def _calculate_v(self):
        """Create diagonal matrix consisting of our predictions of the Poisson distributions
        """
        _lambda0, _lambda1 = self._calculate_lambda()
        _v0 = np.diagflat(_lambda0)
        _v1 = np.diagflat(_lambda1)
        return _v0, _v1

    def _calculate_w(self):
        """Create a diagonal matrix consisting of the difference between our predictions of the 2 Poisson distributions
        with their observed values
        """
        _lambda0, _lambda1 = self._calculate_lambda()
        _w0 = np.diagflat((self.l0 - _lambda0.reshape(-1, 1)) ** 2)
        _w1 = np.diagflat((self.l1 - _lambda1.reshape(-1, 1)) ** 2)
        return _w0, _w1

    def _calculate_robust_covariance(self):
        """Calculate robust variance covariance matrices for our two sets of coefficients
        """
        _v0, _v1 = self._calculate_v()
        _w0, _w1 = self._calculate_w()
        _robust_cov0 = np.linalg.inv(np.dot(np.dot(self._x0.T, _v0), self._x0)) \
                           * np.dot(np.dot(self._x0.T, _w0), self._x0) \
                           * np.linalg.inv(np.dot(np.dot(self._x0.T, _v0), self._x0))
        _robust_cov1 = np.linalg.inv(np.dot(np.dot(self._x1.T, _v1), self._x1)) \
                           * np.dot(np.dot(self._x1.T, _w1), self._x1) \
                           * np.linalg.inv(np.dot(np.dot(self._x1.T, _v1), self._x1))
        return _robust_cov0, _robust_cov1

    def _calculate_robust_standard_errors(self):
        """Calculate robust standard errors for our two sets of coefficients by taking the square root of the diagonal
        values in the variance covariance matrices
        """
        _robust_cov0, _robust_cov1 = self._calculate_robust_covariance()
        _std_error0 = np.sqrt(np.diag(_robust_cov0))
        _std_error1 = np.sqrt(np.diag(_robust_cov1))
        return _std_error0, _std_error1,

    def _calculate_z_values(self):
        """Calculate z statistics for our two sets of coefficients
        """
        _std_error0, _std_error1 = self._calculate_robust_standard_errors()
        _z_values0 = self.lambda_0_coefficients[:, 0] / _std_error0
        _z_values1 = self.lambda_1_coefficients[:, 0] / _std_error1
        return _z_values0, _z_values1

    def _calculate_p_values(self):
        """Calculate p values for our two sets of coefficients
        """
        _z_values0, _z_values1 = self._calculate_z_values()
        _p_values0 = scipy.stats.norm.sf(abs(_z_values0)) * 2
        _p_values1 = scipy.stats.norm.sf(abs(_z_values1)) * 2
        return _p_values0, _p_values1
