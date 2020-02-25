#!/usr/bin/env python
import numpy as np
import pandas as pd


class ArrayUtils:

    @staticmethod
    def convert_to_array(_x):
        if isinstance(_x, np.ndarray) and _x.ndim == 1:
            return _x.reshape(-1, 1)
        elif isinstance(_x, pd.core.series.Series):
            return _x.values.reshape(-1, 1)
        else:
            return _x

    @staticmethod
    def add_intercept(train):
        return np.concatenate((np.ones(len(train))[:, np.newaxis], train), axis=1)

    @staticmethod
    def split_or_duplicate_x(x, add_intercept=True):
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
            x0 = ArrayUtils.convert_to_array(x)
            x1 = ArrayUtils.convert_to_array(x)
        elif len(x) == 2:
            if (
                    isinstance(x[0], np.ndarray)
                    or isinstance(x[0], pd.core.series.Series)
                    or isinstance(x[0], pd.core.frame.DataFrame)
            ):
                x0 = ArrayUtils.convert_to_array(x[0])
            if (
                    isinstance(x[1], np.ndarray)
                    or isinstance(x[1], pd.core.series.Series)
                    or isinstance(x[1], pd.core.frame.DataFrame)
            ):
                x1 = ArrayUtils.convert_to_array(x[1])
            else:
                x0 = x[0]
                x1 = x[1]
        if x0 is None and x1 is None:
            raise ValueError(
                "x must either be an a list of two arrays or a single array"
            )
        if add_intercept:
            x0 = ArrayUtils.add_intercept(x0)
            x1 = ArrayUtils.add_intercept(x1)
        return x0, x1
