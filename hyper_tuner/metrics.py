# utf-8
# Python 3.9
# 2021-04-13


import numpy as np


def MAE(y_true, y_pred):
    """
    Compute MAE loss function.
    """

    return np.mean(np.abs(y_true - y_pred))


def MAPE(y_true, y_pred):
    """
    Compute MAPE loss function.
    """

    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6)))


def MSE(y_true, y_pred):
    """
    Compute MSE loss function.
    """

    return np.mean(np.power(y_true - y_pred, 2))


def RMSE(y_true, y_pred):
    """
    Compute RMSE loss function.
    """

    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))


def RMSLE(y_true, y_pred):
    """
    Compute RMSLE loss function.
    """

    return RMSE(np.log(y_true + 1), np.log(y_pred + 1))


def accuracy(y_true, y_pred):
    """
    Compute Accuracy score function.
    """

    return np.sum(y_true == y_pred) / len(y_true)


def precision(y_true, y_pred):
    """
    Compute Accuracy score function.
    """

    return None


def recall(y_true, y_pred):
    """
    Compute Recall score function.
    """

    return None


def f_score(y_true, y_pred, d=1):
    """
    Compute F_d score function.
    """

    return None
