from typing import Union, Optional, Tuple

import numpy as np
import pandas as pd


def sanitize_input_1d(data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
    """
    Converts 1D array-like data (numpy array, pandas dataframe/series) to a numpy array.

    Parameters
    ----------
    data : array_like
        input data. Needs to be 1D

    Returns
    -------
    array_like
        data as numpy array

    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        # only 1D pandas DataFrame allowed
        if isinstance(data, pd.DataFrame) and len(data.columns) != 1:
            raise ValueError("Only 1D DataFrames allowed!")
        data = np.squeeze(data.values)

    return data


def sanitize_input_nd(
        data: Union[pd.DataFrame, pd.Series, np.ndarray],
        ncols: Optional[Union[int, Tuple[int, ...]]] = None
) -> np.ndarray:
    """
    Converts nD array-like data (numpy array, pandas dataframe/series) to a numpy array.

    Parameters
    ----------
    data : array_like
        input data
    ncols : int or tuple of ints
        number of columns (2nd dimension) the 'data' array should have

    Returns
    -------
    array_like
        data as numpy array
    """

    # ensure tuple
    if isinstance(ncols, int):
        ncols = (ncols,)

    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = np.squeeze(data.values)

    if data.ndim == 1:
        if 1 in ncols:
            return data
        else:
            raise ValueError("Invalid number of columns! Expected one of {}, got 1.".format(ncols))
    elif data.shape[1] not in ncols:
        raise ValueError("Invalid number of columns! Expected one of {}, got {}.".format(ncols, data.shape[1]))
    return data
