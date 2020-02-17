#!/usr/bin/env python
import numpy as np


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

    def calculate_lambda(self):
        _lambda = dict()
        _lambda['1'] = np.exp(np.squeeze(self._x @ self.lambda_1_coefficients))
        _lambda['2'] = np.exp(np.squeeze(self._x @ self.lambda_2_coefficients))
        return _lambda

    def calculate_v(self):
        _lambda = self.calculate_lambda()
        _v = dict()
        _v['1'] = np.diagflat(_lambda['1'])
        _v['2'] = np.diagflat(_lambda['2'])
        return _v

    def calculate_w(self):
        _lambda = self.calculate_lambda()
        _w = dict()
        _w['1'] = np.diagflat((self.l1 - _lambda['1'].reshape(-1, 1)) ** 2)
        _w['2'] = np.diagflat((self.l2 - _lambda['2'].reshape(-1, 1)) ** 2)
        return _w

    def calculate_robust_covariance(self):
        _v = self.calculate_v()
        _w = self.calculate_w()



