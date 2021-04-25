# utf-8
# Python 3.9
# 2021-04-13


import numpy as np


class __Pool:
    """
    Data pool.
    """

    def __init__(self, lib, X, y, **kwargs):
        """
        Initialization.

        Parameters:
            lib (str) - estimator's library name.
            X ([pd.DataFrame|np.array, dim=(n,m)) - features.
            y ([pd.Series|np.array], dim=(n,)) - target.
            **kwargs (dict)
        """

        self.__lib = lib

        if self.__lib == "catboost":
            self.__pool = __import__("catboost").Pool(data=X,
                                                      label=y,
                                                      cat_features=kwargs.get("cat_params", []))
        elif self.__lib == "lightgbm":
            self.__pool = {"X": X, "y": y}


    def __getter__(self, obj, cls):
        return self.__pool


    def __getitem__(self, item):
        return self.__pool.get(item)


    def get_label(self):
        """
        Get target values.

        Returns:
            y ([pd.Series|np.array], dim=(n,)).
        """
        if self.__lib == "catboost":
            return self.__pool.get_label()
        elif self.__lib == "lightgbm":
            return self.__pool["y"]



class __Model:
    """
    ML model.
    """

    def __init__(self, lib, estimator, init_params={}):
        """
        Initialization.

        Parameters:
            lib (str) - estimator's library name.
            estimator (Model) - ML model with sklearn API.
            init_params (dict) - estimator's initialization parameters.
        """

        self.__estimator = estimator(**init_params)
        self.__lib = lib


    def fit(self, train_pool, fit_params={}, **kwargs):
        """
        Train model.

        Parameters:
            train_pool (__Pool) - train data pool.
            fit_params (dict) - parameters for fitting estimator.
        """

        if self.__lib == "catboost":
            self.__estimator.fit(train_pool, **fit_params)
        elif self.__lib == "lightgbm":
            fit_params.update({"categorical_feature": kwargs.get("cat_params", [])})
            self.__estimator.fit(X=train_pool["X"], y=train_pool["y"])


    def predict(self, val_pool, predict_params={}):
        """
        Predict target for validation pool.

        Parameters
            val_pool (__Pool) - validation data pool.
            predict_params (dict) - parameters for estimator's prediction.

        Returns:
            y_pred ([pd.Series|np.array], dim=(n,)) - predict values.
        """

        if self.__lib == "catboost":
            return self.__estimator.predict(val_pool)
        elif self.__lib == "lightgbm":
            return self.__estimator.predict(X=val_pool["X"])



def cross_val_loss(lib, estimator, X, y, cv,
                   loss_fun, agg_fun,
                   init_params, fit_params, predict_params,
                   cat_params):
    """
    Compute cross-validation score.

    Parameters:
        lib (str) - estimator's library name.
        estimator (Model) - model with sklearn API.
        X ([pd.DataFrame|np.array, dim=(n,m)) - features.
        y ([pd.Series|np.array], dim=(n,)) - target.
        loss_fun (fun) - loss function.
        agg_fun (fun) - losses aggregation function.
        init_params (dict) - estimator's initialization parameters.
        fit_params (dict) - parameters for fitting estimator.
        predict_params (dict) - parameters for estimator's prediction.
        cat_params (list) - list of categorical parameters.

    Returns:
        loss (float) - agg loss value.
    """

    model = __Model(lib=lib,
                    estimator=estimator,
                    init_params=init_params)

    cv = KFold(n_splits=5)
    losses = np.array([], dtype=np.float64)

    for train_index, val_index in cv.split(X, y):
        train_pool = __Pool(lib, X[train_index], y[train_index], cat_params=cat_params)
        val_pool = __Pool(lib, X[val_index], y[val_index], cat_params=cat_params)

        model.fit(train_pool=train_pool,
                  fit_params=fit_params,
                  cat_params=cat_params)

        y_true = val_pool.get_label()
        y_pred = model.predict(val_pool=val_pool,
                               predict_params=predict_params)

        loss = loss_fun(y_true, y_pred)
        losses = np.append(losses, loss)


    return agg_fun(losses)





