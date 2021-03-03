from typing import Union

import numpy as np
import pandas as pd


def se(data: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    return np.std(data, ddof=1) / np.sqrt(len(data))


def mean_se(data: pd.DataFrame):
    return data.agg([np.mean, se])
