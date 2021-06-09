"""Module with utility functions to handle sleep data."""
from typing import Sequence, Optional

import numpy as np
import pandas as pd


# TODO default split at 6pm because that's the time of day where the probability that people are sleeping is the lowest
def split_nights(data: pd.DataFrame, diff_hours: Optional[int] = 12) -> Sequence[pd.DataFrame]:
    """Split continuous data into individual nights.

    This function splits data into individual nights when two successive timestamps differ more than the threshold
    provided by ``diff_hours``.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        input data
    diff_hours : int, optional
        minimum difference between two successive timestamps required to split data into individual nights

    Returns
    -------
    list
        list of dataframes split into individual nights

    """
    idx_split = np.where(np.diff(data.index, prepend=data.index[0]) > pd.Timedelta(diff_hours, "hours"))[0]
    list_nights = np.split(data, idx_split)
    return list_nights
