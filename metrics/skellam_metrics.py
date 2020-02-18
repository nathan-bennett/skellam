#!/usr/bin/env python
import numpy as np
import scipy


class SkellamMetrics:

    def __init__(self, x, y, y_hat, model, l1, l2):
        self._x = x
        self._y = y
        self._y_hat = y_hat
        self.model = model
        self.l1 = l1
        self.l2 = l2
        self.coeff_size = len(model) // 2
        self.lambda_1_coefficients = model[0: self.coeff_size].reshape(-1, 1)
        self.lambda_2_coefficients = model[self.coeff_size:].reshape(-1, 1)

    def sse(self):
        return ((self._y - self._y_hat)**2).sum()

    def _y_bar(self):
        return self._y.mean()

    def sst(self):
        return ((self._y - self._y_bar())**2).sum()

    def r2(self):
        sse_sst = self.sse()/self.sst()
        return 1-sse_sst

    def _calculate_lambda(self):
        _lambda = dict()
        _lambda['1'] = np.exp(np.squeeze(self._x @ self.lambda_1_coefficients))
        _lambda['2'] = np.exp(np.squeeze(self._x @ self.lambda_2_coefficients))
        return _lambda

    def _calculate_v(self):
        _lambda = self._calculate_lambda()
        _v = dict()
        _v['1'] = np.diagflat(_lambda['1'])
        _v['2'] = np.diagflat(_lambda['2'])
        return _v

    def _calculate_w(self):
        _lambda = self._calculate_lambda()
        _w = dict()
        _w['1'] = np.diagflat((self.l1 - _lambda['1'].reshape(-1, 1)) ** 2)
        _w['2'] = np.diagflat((self.l2 - _lambda['2'].reshape(-1, 1)) ** 2)
        return _w

    def _calculate_robust_covariance(self):
        _v = self._calculate_v()
        _w = self._calculate_w()
        _robust_cov = dict()
        _robust_cov['1'] = np.linalg.inv(np.dot(np.dot(self._x.T, _v['1']), self._x)) \
                           * np.dot(np.dot(self._x.T, _w['1']), self._x) \
                           * np.linalg.inv(np.dot(np.dot(self._x.T, _v['1']), self._x))
        _robust_cov['2'] = np.linalg.inv(np.dot(np.dot(self._x.T, _v['2']), self._x)) \
                           * np.dot(np.dot(self._x.T, _w['2']), self._x) \
                           * np.linalg.inv(np.dot(np.dot(self._x.T, _v['2']), self._x))
        return _robust_cov

    def _calculate_robust_standard_errors(self):
        _std_error = dict()
        _robust_cov = self._calculate_robust_covariance()
        _std_error['1'] = np.sqrt(np.diag(_robust_cov['1']))
        _std_error['2'] = np.sqrt(np.diag(_robust_cov['2']))
        return _std_error

    def _calculate_z_values(self):
        _std_error = self._calculate_robust_standard_errors
        _z_values = dict()
        _z_values['1'] = self.lambda_1_coefficients / _std_error['1']
        _z_values['1'] = self.lambda_2_coefficients / _std_error['2']
        return _z_values

    def _calculate_p_values(self):
        _z_values = self._calculate_z_values()
        _p_values = dict()
        _p_values['1'] = scipy.stats.norm.sf(abs(_z_values['1'])) * 2
        _p_values['2'] = scipy.stats.norm.sf(abs(_z_values['2'])) * 2
        return _p_values


