from typing import Dict, Union, Optional
from biopsykit.sleep.sleep_wake_detection.algorithms._base import _SleepWakeBase
from biopsykit.utils.array_handling import sanitize_input_1d
from biopsykit.utils.array_handling import sliding_window
from biopsykit.utils.rescore import rescore
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


    def fit(self):
        pass


    def predict(self, data: Union[pd.DataFrame, np.array], rescore_data: Optional[bool] = True) -> Union[np.array, pd.DataFrame]:
        """
        Perform the sleep/wake score prediction.

        Parameters
        ----------
        data : pd.DataFrame
            pandas dataframe of activity index values.

        Returns
        -------
        np.array
            predictions with sleep = True and wake = False
        """
        index = None
        if isinstance(data, pd.DataFrame):
            index = data.index
            data = sanitize_input_1d(data)

        window_past = 6
        window_mean = 11
        window_center = 11

        #mean = (self._rolling_window(data, window_mean, window_mean - 1)).mean(1)
        mean = sliding_window(data,window_samples=window_mean,overlap_samples=window_mean-1).mean(1)
        #nat = self._rolling_window(data, window_center, window_center - 1)
        nat = sliding_window(data,window_samples=window_center,overlap_samples=window_center-1)
        nat = np.logical_and(nat < 100, nat > 50)
        nat = np.sum(nat, axis=1)
        #std = (self._rolling_window(data, window_past, window_past - 1)).std(1)[:-5]
        std = sliding_window(data,window_samples=window_past,overlap_samples=window_past-1).std(1)[:-5]
        locAct = np.log(data + 1)[5:-5]
        score = 7.601 - 0.065 * mean - 0.056 * std - 0.0703 * locAct - 1.08 * nat

        score[score >= 0] = 1       #sleep = 1
        score[score < 0] = 0        #wake = 0

        score = np.pad(np.asarray(score),(5), 'constant')

        if rescore_data:
            score = rescore(score)

        if index is not None:
            score = pd.DataFrame(score, index=index, columns=["sleep_wake"])

        return score


