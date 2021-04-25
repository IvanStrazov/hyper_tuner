

from sklearn.model_selection import KFold
from hyperopt import hp

OBJECTIVE = "classification" # "regression" / "classification"
LIBRARY = "lightgbm" # "lightgbm" / "catboost

CV = KFold


ESTIMATOR = {
    "regression": {
        "catboost": __import__("catboost").CatBoostRegressor,
        "lightgbm": __import__("lightgbm").LGBMRegressor
    },
    "classification": {
        "catboost": __import__("catboost").CatBoostClassifier,
        "lightgbm": __import__("lightgbm").LGBMClassifier
    }
}.get(OBJECTIVE).get(LIBRARY)

SPACE = {
    "catboost": {
        "depth": hp.quniform("depth", 3, 16, 1),
        "iterations": hp.quniform("iterations", 50, 1000, 50),
        "learning_rate": hp.uniform("learning_rate", 1e-1, 5e-1),
        "min_child_samples": hp.quniform("min_child_samples", 1, 300, 3),
        "reg_lambda": hp.uniform("reg_lambda", 0, 10),
        "random_strength": hp.lognormal("random_strength", 1e-9, 1)
    },
    "lightgbm": {
        "max_depth": hp.quniform("max_depth", 3, 26, 1),
        "n_estimators": hp.quniform("n_estimators", 50, 1000, 50),
        "learning_rate": hp.uniform("learning_rate", 0.00001, 0.5),
        "min_child_samples": hp.quniform("min_child_samples", 1, 300, 3),
        "reg_alpha": hp.uniform("reg_alpha", 0, 1),
        "reg_lambda": hp.uniform("reg_lambda", 0, 1),
        "num_leaves": hp.quniform("num_leaves", 2, 110, 3)
    }
}.get(LIBRARY)


COMMON_PARAMS = {
    "catboost": {
        "random_state": 42,
        "task_type": "GPU", # "CPU"/"GPU"
        "objective": "RMSE",
        "eval_metric": "RMSE",
        "od_type": "Iter",
        "od_wait": 20,
        "use_best_model": True
    },
    "lightgbm": {
        "random_state": 42,
        "n_jobs": -1,
        "objective": "RMSE"
    }
}.get(LIBRARY)

FIT_PARAMS = {
    "catboost": {
        "verbose": 0
    },
    "lightgbm": {
        "verbose": 0
    }
}.get(LIBRARY)

PREDICT_PARAMS = {
    "catboost": {},
    "lightgbm": {}
}.get(LIBRARY)

INT_PARAMS = {
    "catboost": ["iterations", "depth", "min_child_samples"],
    "lightgbm": ["max_depth", "n_estimators", "min_child_samples", "num_leaves"]
}.get(LIBRARY)
