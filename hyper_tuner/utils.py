# utf-8
# Python 3.9
# 2021-04-25


import pandas as pd


def to_numpy(data):
    """
    Transform dataset to numpy array.

    Parameters:
        data ([pd.DataFrame|ps.Series|np.array]) - input data with unknown datatype.

    Returns:
        data (np.array) - output data as numpy array.
    """

    if type(data) in (pd.DataFrame, pd.Series):
        return data.values
    else:
        return data
