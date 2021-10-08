"""Sleep/Wake detection using the *Sadeh Algorithm*."""
import numpy as np
import pandas as pd

from biopsykit.sleep.sleep_wake_detection.algorithms._base import _SleepWakeBase
from biopsykit.sleep.sleep_wake_detection.utils import rescore
from biopsykit.utils._types import arr_t
from biopsykit.utils.array_handling import sanitize_input_1d, sliding_window
from biopsykit.utils.datatype_helper import SleepWakeDataFrame, _SleepWakeDataFrame


class Sadeh(_SleepWakeBase):
    """Class representing the *Sadeh Algorithm* for sleep/wake detection based on activity counts."""

    def __init__(self, **kwargs):  # pylint:disable=useless-super-delegation
        """Class representing the *Sadeh Algorithm* for sleep/wake detection based on activity counts.

        The *Sadeh Algorithm* runs sleep wake detection on epoch level activity data. Epochs are 1 minute long and
        activity is represented by an activity index which comes from Actigraph data or from raw acceleration data
        converted into activity index data.

        References
        ----------
        add reference

        """
        super().__init__(**kwargs)

    def fit(self, data: arr_t, **kwargs):
        """Fit sleep/wake detection algorithm to input data.

        .. note::
            Algorithms that do not have to (re)fit a ML model before sleep/wake prediction, such as rule-based
            algorithms, will internally bypass this method as the ``fit`` step is not needed.

        Parameters
        ----------
        data : array_like
            input data

        """
        return

    def predict(self, data: arr_t, **kwargs) -> SleepWakeDataFrame:
        """Apply sleep/wake prediction algorithm on input data.

        Parameters
        ----------
        data : array_like
            array with activity index values
        **kwargs :
            additional arguments to be passed to the algorithm for prediction, such as:

            * ``rescore_data`` (``bool``):
              ``True`` to apply Webster's rescoring rules to the sleep/wake predictions, ``False`` otherwise.
              Default: ``True``
            * ``epoch_length`` (``int``):
              activity data epoch lengths in seconds, i.e. Epoch lengths are usually 30 or 60 seconds.
              Default: 60

        Returns
        -------
        :obj:`~biopsykit.utils.datatype_helper.SleepWakeDataFrame`
            dataframe with sleep/wake predictions

        """
        index = None
        rescore_data: bool = kwargs.get("rescore_data", True)
        epoch_length: bool = kwargs.get("epoch_length", 60)

        if isinstance(data, pd.DataFrame):
            index = data.index
            data = sanitize_input_1d(data)

        window_past = 6
        window_mean = 11
        window_center = 11

        mean = sliding_window(data, window_samples=window_mean, overlap_samples=window_mean - 1).mean(1)
        nat = sliding_window(data, window_samples=window_center, overlap_samples=window_center - 1)
        nat = np.sum(np.logical_and(nat < 100, nat > 50), axis=1)
        std = sliding_window(data, window_samples=window_past, overlap_samples=window_past - 1).std(1)[:-5]
        loc_act = np.log(data + 1)[5:-5]
        score = 7.601 - 0.065 * mean - 0.056 * std - 0.703 * loc_act - 1.08 * nat

        score[score >= 0] = 1  # sleep = 1
        score[score < 0] = 0  # wake = 0

        score = np.pad(np.asarray(score), (5), "constant")
        if rescore_data:
            score = rescore(score, epoch_length=epoch_length)

        if index is not None:
            score = pd.DataFrame(score, index=index, columns=["sleep_wake"])

        return _SleepWakeDataFrame(score)
