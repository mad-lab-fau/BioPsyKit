"""Module with a set of functions useful for computing."""
from typing import Union

import numpy as np
import pandas as pd


def se(data: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    """Compute standard error (SE).

    .. note::
        For computing the standard error the *corrected* sample standard deviation (``ddof = 1``) is used

    Parameters
    ----------
    data : array_like
        input data

    Returns
    -------
    array_like
        standard error of data

    """
    return np.std(data, ddof=1) / np.sqrt(len(data))


def mean_se(data: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and standard error from a dataframe.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        input data

    Returns
    -------
    :class:`~pandas.DataFrame`
        mean and standard error of data

    """
    return data.agg([np.mean, se])
