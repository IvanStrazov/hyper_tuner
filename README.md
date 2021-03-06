# Hyperparameter's tuner for ML models.

## Simple example

```
from hyper_tuner.core import make_hyperopt
import config

X, y = ...

tuner_pipe = make_hyperopt(estimator=config.ESTIMATOR,
                           X=X, y=y,
                           cv=config.CV,
                           space=config.SPACE, common_params=config.COMMON_PARAMS,
                           fit_params=config.FIT_PARAMS, predict_params=config.PREDICT_PARAMS,
                           cat_params=[], int_params=config.INT_PARAMS,
                           max_evals=2)
100%|██████████████████████████████████████████████████| 2/2 [00:06&lt;00:00,  3.25s/trial, best loss: 14.657120473611053]

print(tuner_pipe.best_parameters)
>> {'learning_rate': 0.16406423368959028, 'max_depth': 10,
    'min_child_samples': 60, 'n_estimators': 500,
    'num_leaves': 6, 'reg_alpha': 0.6174653695343534,
    'reg_lambda': 0.6614076732764504}
```

## References

Needed packages can be installed via `pip install -r requirements.txt`

