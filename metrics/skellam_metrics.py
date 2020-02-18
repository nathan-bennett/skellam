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
        x0 = None
        x1 = None
        if isinstance(x, np.ndarray) or isinstance(x, pd.core.series.Series) or isinstance(x, pd.core.frame.DataFrame):
            x0, x1 = x, x
        elif len(x) == 2:
            if isinstance(x[0], np.ndarray) or isinstance(x[0], pd.core.series.Series) or isinstance(x, pd.core.frame.DataFrame):
                x0 = self.convert_to_array(x[0])
            if isinstance(x[1], np.ndarray) or isinstance(x[1], pd.core.series.Series) or isinstance(x, pd.core.frame.DataFrame):
                x1 = self.convert_to_array(x[1])
            else:
                x0 = x[0]
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
        _lambda = dict()
        _lambda['0'] = self.convert_to_array(np.exp(np.squeeze(self._x0 @ self.lambda_0_coefficients)))
        _lambda['1'] = self.convert_to_array(np.exp(np.squeeze(self._x1 @ self.lambda_1_coefficients)))
        return _lambda

    def _calculate_v(self):
        """Create diagonal matrix consisting of our predictions of the Poisson distributions
        """
        _lambda = self._calculate_lambda()
        _v = dict()
        _v['0'] = np.diagflat(_lambda['0'])
        _v['1'] = np.diagflat(_lambda['1'])
        return _v

    def _calculate_w(self):
        """Create a diagonal matrix consisting of the difference between our predictions of the 2 Poisson distributions
        with their observed values
        """
        _lambda = self._calculate_lambda()
        _w = dict()
        _w['0'] = np.diagflat((self.l0 - _lambda['0'].reshape(-1, 1)) ** 2)
        _w['1'] = np.diagflat((self.l1 - _lambda['1'].reshape(-1, 1)) ** 2)
        return _w

    def _calculate_robust_covariance(self):
        """Calculate robust variance covariance matrices for our two sets of coefficients
        """
        _v = self._calculate_v()
        _w = self._calculate_w()
        _robust_cov = dict()
        _robust_cov['0'] = np.linalg.inv(np.dot(np.dot(self._x0.T, _v['0']), self._x0)) \
                           * np.dot(np.dot(self._x0.T, _w['0']), self._x0) \
                           * np.linalg.inv(np.dot(np.dot(self._x0.T, _v['0']), self._x0))
        _robust_cov['1'] = np.linalg.inv(np.dot(np.dot(self._x1.T, _v['1']), self._x1)) \
                           * np.dot(np.dot(self._x1.T, _w['1']), self._x1) \
                           * np.linalg.inv(np.dot(np.dot(self._x1.T, _v['1']), self._x1))
        return _robust_cov

    def _calculate_robust_standard_errors(self):
        """Calculate robust standard errors for our two sets of coefficients by taking the square root of the diagonal
        values in the variance covariance matrices
        """
        _robust_cov = self._calculate_robust_covariance()
        _std_error = dict()
        _std_error['0'] = np.sqrt(np.diag(_robust_cov['0']))
        _std_error['1'] = np.sqrt(np.diag(_robust_cov['1']))
        return _std_error

    def _calculate_z_values(self):
        """Calculate z statistics for our two sets of coefficients
        """
        _std_error = self._calculate_robust_standard_errors()
        _z_values = dict()
        _z_values['0'] = self.lambda_0_coefficients[:, 0] / _std_error['0']
        _z_values['1'] = self.lambda_1_coefficients[:, 0] / _std_error['1']
        return _z_values

    def _calculate_p_values(self):
        """Calculate p values for our two sets of coefficients
        """
        _z_values = self._calculate_z_values()
        _p_values = dict()
        _p_values['0'] = scipy.stats.norm.sf(abs(_z_values['0'])) * 2
        _p_values['1'] = scipy.stats.norm.sf(abs(_z_values['1'])) * 2
        return _p_values
