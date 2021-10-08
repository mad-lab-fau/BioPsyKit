"""Sleep/Wake detection using the *Sazonov Algorithm*."""
import numpy as np
import pandas as pd
from scipy.special import expit

from biopsykit.sleep.sleep_wake_detection.algorithms._base import _SleepWakeBase
from biopsykit.sleep.sleep_wake_detection.utils import rescore
from biopsykit.utils._types import arr_t
from biopsykit.utils.array_handling import sanitize_input_1d, sliding_window
from biopsykit.utils.datatype_helper import SleepWakeDataFrame, _SleepWakeDataFrame


class Sazonov(_SleepWakeBase):
    """Class representing the *Sazonov Algorithm* for sleep/wake detection based on activity counts."""

    def __init__(self, **kwargs):  # pylint:disable=useless-super-delegation
        """Class representing the *Sazonov Algorithm* for sleep/wake detection based on activity counts.

        The *Sazonov Algorithm* runs sleep wake detection on epoch level activity data. Epochs are 1 minute long and
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
            input data with activity index values
        **kwargs :
            additional arguments to be passed to the algorithm for prediction, such as:

            * ``rescore_data`` (``bool``):
              ``True`` to apply Webster's rescoring rules to the sleep/wake predictions, ``False`` otherwise.
              Default: ``True``
            * ``epoch_length`` (``int``):
              activity data epoch lengths in seconds, i.e. Epoch lengths are usually 30 or 60 seconds.
              Default: 30

        Returns
        -------
        :obj:`~biopsykit.utils.datatype_helper.SleepWakeDataFrame`
            dataframe with sleep/wake predictions

        """
        index = None
        rescore_data: bool = kwargs.get("rescore_data", True)
        epoch_length: bool = kwargs.get("epoch_length", 30)

        if isinstance(data, pd.DataFrame):
            index = data.index
            data = sanitize_input_1d(data)

        w0 = data
        w1 = np.pad(np.max(sliding_window(data, window_samples=2, overlap_samples=1), axis=1), (1, 0))
        w2 = np.pad(np.max(sliding_window(data, window_samples=3, overlap_samples=2), axis=1), (2, 0))
        w3 = np.pad(np.max(sliding_window(data, window_samples=4, overlap_samples=3), axis=1), (3, 0))
        w4 = np.pad(np.max(sliding_window(data, window_samples=5, overlap_samples=4), axis=1), (4, 0))

        scores = np.array(1.727 - 0.256 * w0 - 0.154 * w1 - 0.136 * w2 - 0.140 * w3 - 0.176 * w4)
        scores = expit(scores)

        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0

        if rescore_data:
            scores = rescore(scores, epoch_length=epoch_length)

        if index is not None:
            scores = pd.DataFrame(scores, index=index, columns=["sleep_wake"])

        return _SleepWakeDataFrame(scores)
