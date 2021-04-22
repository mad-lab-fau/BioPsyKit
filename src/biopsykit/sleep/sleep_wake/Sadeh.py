from typing import Union
from biopsykit.sleep.sleep_wake.base import _SleepWakeBase
from numpy.lib.stride_tricks import as_strided
import numpy as np
import pandas as pd


class Sadeh(_SleepWakeBase):
    """
    Runs sleep wake detection on epoch level activity data. Epochs are 1 minute long and activity is represented
    by an activity index.
    """

    def __init__(self):
        """
        Create an instance of the Sadeh Algorithm class for sleep/wake detection.
        """
        pass

    def predict(self, data: Union[pd.DataFrame, np.array]) -> Union[np.array, pd.DataFrame]:
        """
        Perform the sleep/wake score prediction.

        Parameters
        ----------
        data : pd.DataFrame
            pandas dataframe of activity index values.

        Returns
        -------
        np.array
            predictions
        """
        index = None
        if isinstance(data, pd.DataFrame):
            index = data.index
            data = sanitize_input_1d(data)

        window_past = 6
        window_mean = 11
        window_center = 11

        mean = (self._rolling_window(data, window_mean, window_mean - 1)).mean(1)
        nat = self._rolling_window(data, window_center, window_center - 1)
        nat = np.logical_and(nat < 100, nat > 50)
        nat = np.sum(nat, axis=1)
        std = (self._rolling_window(data, window_past, window_past - 1)).std(1)[:-5]
        locAct = np.log(data + 1)[5:-5]
        score = 7.601 - 0.065 * mean - 0.056 * std - 0.0703 * locAct - 1.08 * nat

        classification = (score > 0)

        if index is not None:
            classification = pd.DataFrame(classification, index=index, columns=["sleep_wake"])

        return classification

    @staticmethod
    def _rolling_window(array, window, overlap):
        """
        Usage of numpy stride tricks to develop a rolling window method.

        :param array: array of activity index values
        :param window: window size of rolling window
        :param overlap overlap of rolling window
        """
        window_step = window - overlap
        new_shape = array.shape[:-1] + ((array.shape[-1] - overlap) // window_step, window)
        new_strides = (array.strides[:-1] + (window_step * array.strides[-1],) + array.strides[-1:])
        overlap_matrix = as_strided(array, shape=new_shape, strides=new_strides)

        return overlap_matrix
