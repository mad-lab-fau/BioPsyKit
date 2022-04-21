"""Functions to handle multicollinearity in data."""

from typing import Optional

import numpy as np
import pandas as pd


def remove_multicollinearity_correlation(data: pd.DataFrame, threshold: Optional[float] = 0.8) -> pd.DataFrame:
    """Remove features with multicollinearity based on cross-correlation coefficient.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Input data with features to check for multicollinearity.
    threshold : float, optional
        Cross-correlation coefficient threshold. Features with a correlation coefficient above this value will be
        removed. Default: 0.8

    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe without features with high multicollinearity.

    """
    corr_data = pd.DataFrame(np.triu(np.abs(data.corr())), columns=data.columns)

    multicoll_columns = np.logical_and(corr_data >= threshold, corr_data < 1.0).any()
    return data.loc[:, ~multicoll_columns]
