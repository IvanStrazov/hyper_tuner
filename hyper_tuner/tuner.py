# utf-8
# Python 3.9
# 2021-04-13

import functools
from hyperopt import fmin, tpe, STATUS_OK, Trials
from hyper_tuner import model



class Tuner:
    """
    Search best hyperparameters for model.
    """

    def __init__(self, lib, estimator,
                 X, y,
                 space, common_params={},
                 fit_params={}, predict_params={},
                 cat_params=[], int_params=[]):
        """
        Initialization.

        Parameters:
            lib (str) - estimator's library name.
            estimator (Model) - model with sklearn API.
            X (catboost.Pool, dim=(n,m)) - features.
            y (catboost.Pool, dim=(k,m)) - target values.
            space (dict) - space for searching hyperparameters.
            common_params (dict) - common model's parameters.
            fit_params (dict) - parameters for fitting estimator.
            predict_params (dict) - parameters for estimator's prediction.
            cat_params (list) - list of categorical parameters.
            int_params (list) - list of integer parameters.
        """

        self.__lib = lib
        self.__estimator = estimator
        self.__X = X
        self.__y = y
        self.__space = space
        self.__init_params = common_params
        self.__fit_params = fit_params
        self.__predict_params = predict_params
        self.__cat_params = cat_params
        self.__int_params = int_params


    def __params2int(self, params):
        """
        Transform some hyperopt formats to integer.

        Parameters:
            params (dict) - model parameters.
        """

        for par in self.__int_params:
            params[par] = int(params[par])

        return params


    def __score(self, model_params, loss_fun, agg_fun):
        """
        Compute score over dictionary of parameters.

        Parameters:
            model_params (dict) - model parameters.
            loss_fun (fun) - loss function.
            agg_fun (fun) - losses aggregation function.
        """

        model_params = self.__params2int(model_params)
        self.__init_params.update(model_params)

        loss = model.cross_val_loss(lib=self.__lib, estimator=self.__estimator,
                                    X=self.__X, y=self.__y,
                                    loss_fun=loss_fun, agg_fun=agg_fun,
                                    init_params=self.__init_params,
                                    fit_params=self.__fit_params,
                                    predict_params=self.__predict_params,
                                    cat_params=self.__cat_params)

        return {"loss": loss, "status": STATUS_OK}


    def make_tune(self, max_evals, loss_fun, agg_fun):
        """
        Make searching of hyperparameters.

        Parameters:
            max_evals (int) - max number of iterations for searching hyperparameters.
            loss_fun (fun) - loss function.
            agg_fun (fun) - losses aggregation function.
        """

        self.trials = Trials()
        best = fmin(functools.partial(self.__score, loss_fun=loss_fun, agg_fun=agg_fun),
                    space=self.__space,
                    trials=self.trials,
                    algo=tpe.suggest,
                    max_evals=max_evals)
        self.best_parameters = self.__params2int(best)
