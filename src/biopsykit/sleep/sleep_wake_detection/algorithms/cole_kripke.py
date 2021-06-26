from typing import Optional, Union

import numpy as np
import pandas as pd

from biopsykit.utils.array_handling import sanitize_input_1d
from biopsykit.sleep.sleep_wake_detection.algorithms._base import _SleepWakeBase


class ColeKripke(_SleepWakeBase):
    """
    Runs sleep wake detection on epoch level activity data. Epochs are 1 minute long and activity is represented
    by an activity index.
    """

    scale_factor: float = 0.193125

    def __init__(self, **kwargs):
        """
        Creates a new instance of the Cole/Kripke Algorithm class for sleep/wake detection.

        Parameters
        ----------
        scale_factor : float
            scale factor to use for the predictions (default corresponds to scale factor optimized for use with
            the activity index, if other activity measures are desired the scale factor can be modified or optimized.)
            The recommended range for the scale factor is between 0.1 and 0.25 depending on the sensitivity to activity
            desired, and possibly the population being observed.

        """
        self.scale_factor = kwargs.get("scale_factor", self.scale_factor)

    def predict(self, data: Union[pd.DataFrame, np.array], **kwargs) -> Union[np.array, pd.DataFrame]:
        """
        Performs the sleep/wake score prediction.

        Parameters
        ----------
        data : pd.DataFrame
            pandas dataframe of activity index values

        Returns
        -------
        np.array
            rescored predictions
        """

        index = None
        if isinstance(data, pd.DataFrame):
            index = data.index
            data = sanitize_input_1d(data)

        # ensure numpy
        sf = np.array(self.scale_factor)
        kernel = sf * np.array([4.64, 6.87, 3.75, 5.07, 16.19, 5.84, 4.024, 0.00, 0.00])[::-1]
        scores = np.convolve(data, kernel, "same")

        scores[scores >= 1] = 99  # wake = 0
        scores[scores < 1] = 1  # sleep = 1
        scores[scores == 99] = 0  # wake = 0

        # rescore the original predictions
        scores = self._rescore(scores)

        if index is not None:
            scores = pd.DataFrame(scores, index=index, columns=["sleep_wake"])
        return scores

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
