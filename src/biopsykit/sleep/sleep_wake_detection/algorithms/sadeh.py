from typing import Union
from biopsykit.sleep.sleep_wake_detection.algorithms._base import _SleepWakeBase
#from numpy.lib.stride_tricks import as_strided
from biopsykit.utils.array_handling import sanitize_input_1d
from biopsykit.utils.array_handling import sliding_window
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

        score[score >= 0] = 0
        score[score < 0] = 1

        score = np.asarray(score)
        score = self._rescore(score)

        if index is not None:
            score = pd.DataFrame(classification, index=index, columns=["sleep_wake"])

        return score


    @staticmethod
    def _rescore(predictions: np.array) -> np.array:
        """
        Application of Webster's rescoring rules as described in the Cole-Kripke paper.

        :param predictions: array of predictions
        :return: rescored predictions
        """
        rescored = predictions.copy()

        # rules a through c
        wake_bin = 0
        for t in range(len(rescored)):
            if rescored[t] == 1:
                wake_bin += 1
            else:
                if 15 <= wake_bin:
                    # rule c: at least 15 minutes of wake, next 4 minutes of sleep get rescored
                    rescored[t : t + 4] = 1.0
                elif 10 <= wake_bin < 15:
                    # rule b: at least 10 minutes of wake, next 3 minutes of sleep get rescored
                    rescored[t : t + 3] = 1.0
                elif 4 <= wake_bin < 10:
                    # rule a: at least 4 minutes of wake, next 1 minute of sleep gets rescored
                    rescored[t] = 1.0
                wake_bin = 0

        # rule d/e: 6/10 minutes or less of sleep surrounded by at least 10/20 minutes of wake on each side get rescored
        sleep_rules = [6, 10]
        wake_rules = [10, 20]

        for sleep_thres, wake_thres in zip(sleep_rules, wake_rules):
            sleep_bin = 0
            start_ind = 0
            for t in range(wake_thres, len(rescored) - wake_thres):
                if rescored[t] == 0:
                    sleep_bin += 1
                    if sleep_bin == 1:
                        start_ind = t
                else:
                    sum1 = np.sum(rescored[start_ind - wake_thres : start_ind])
                    sum2 = np.sum(rescored[t : t + wake_thres])
                    if 0 < sleep_bin <= sleep_thres and sum1 == wake_thres and sum2 == wake_thres:
                        rescored[start_ind:t] = 1.0
                sleep_bin = 0

        # wake phases of 1 minute, surrounded by sleep, get rescored
        for t in range(1, len(rescored) - 1):
            if rescored[t] == 1 and rescored[t - 1] == 0 and rescored[t + 1] == 0:
                rescored[t] = 0

        return rescored


