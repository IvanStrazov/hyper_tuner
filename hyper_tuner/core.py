# utf-8
# Python 3.9
# 2021-04-13


import numpy as np
from hyper_tuner import tuner
from hyper_tuner import metrics


def make_hyperopt(estimator,
                  X, y,
                  space, common_params={},
                  fit_params={}, predict_params={},
                  cat_params=[], int_params=[],
                  max_evals=50,
                  loss_fun=metrics.RMSE, agg_fun=np.mean) -> tuner.Tuner:
    """
    Search best hyperparameters with optimization over minimize loss function.

    Parameters:
        estimator (Model) - model with sklearn API.
        X ([pd.DataFrame|np.array, dim=(n,m)) - features.
        y ([pd.Series|np.array], dim=(n,)) - target.
        space (dict) - space for searching hyperparameters.
        common_params (dict) - common model's parameters.
        fit_params (dict) - parameters for fitting estimator.
        predict_params (dict) - parameters for estimator's prediction.
        cat_params (list) - list of categorical parameters.
        int_params (list) - list of integer parameters.
        max_evals (int>0, default=50) - max number of iterations for searching hyperparameters.
        loss_fun (fun, default=metrics.RMSE) - loss function.
        agg_fun (fun, default=np.mean) - losses aggregation function.

    Returns:
        tuner_pipe (tuner.Tuner)
    """

    estimator_name = repr(estimator)
    if "catboost" in estimator_name:
        lib = "catboost"
    elif "lightgbm" in estimator_name:
        lib = "lightgbm"

    tuner_pipe = tuner.Tuner(lib=lib, estimator=estimator,
                             X=X, y=y,
                             space=space, common_params=common_params,
                             cat_params=cat_params, int_params=int_params)
    tuner_pipe.make_tune(max_evals=max_evals, loss_fun=loss_fun, agg_fun=agg_fun)

    return tuner_pipe
