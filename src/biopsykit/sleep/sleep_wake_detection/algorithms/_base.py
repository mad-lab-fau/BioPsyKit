from typing import Union

import pandas as pd
import numpy as np


class _SleepWakeBase:
    def __init__(self, **kwargs):
        raise NotImplementedError("Needs to be implemented by child class.")

    def fit(self, data: Union[pd.DataFrame, np.array], **kwargs) -> Union[np.array, pd.DataFrame]:
        raise NotImplementedError("Needs to be implemented by child class.")

    def predict(self, data: Union[pd.DataFrame, np.array], **kwargs) -> Union[np.array, pd.DataFrame]:
        raise NotImplementedError("Needs to be implemented by child class.")
