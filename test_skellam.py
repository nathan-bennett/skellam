"""Tests for `skellam` package."""
import pytest
import numpy as np
from skellam_reg import SkellamRegression

np.random.seed(123)
y1 = np.random.poisson(18, 1000)
y2 = np.random.poisson(14, 1000)

y = y1 - y2

x = np.ones(1000)


def test_model():
    model = SkellamRegression(x,
                              y,
                              y1,
                              y2,
                              add_intercept=True)

    model.train()

    predictions = model.predict(x.reshape(-1, 1))
    assert isinstance(predictions, np.ndarray), "Created predictions"

    assert isinstance(model.model_performance().r2(), float), "Calculated r2"
    assert isinstance(model.model_performance().adjusted_r2(), float), "Calculated adjusted r2"
    assert isinstance(model.model_performance().aic(), float), "Calculated aic"
    assert isinstance(model.model_performance().bic(), float), "Calculated bic"
