"""Sleep/Wake detection using the *Scripps Clinic Algorithm*."""
import numpy as np
import pandas as pd

from biopsykit.sleep.sleep_wake_detection.algorithms._base import _SleepWakeBase
from biopsykit.sleep.sleep_wake_detection.utils import rescore
from biopsykit.utils._types import arr_t
from biopsykit.utils.array_handling import sanitize_input_1d
from biopsykit.utils.datatype_helper import SleepWakeDataFrame, _SleepWakeDataFrame


class ScrippsClinic(_SleepWakeBase):
    """Class representing the *Scripps Clinic Algorithm* for sleep/wake detection based on activity counts."""

    def __init__(self, **kwargs):
        """Class representing the *Scripps Clinic Algorithm* for sleep/wake detection based on activity counts.

        The *Scripps Clinic Algorithm* runs sleep wake detection on epoch level activity data. Epochs are 1 minute
        long and activity is represented by an activity index which comes from Actigraph data or from raw acceleration
        data converted into activity index data.

        Parameters
        ----------
        scale_factor : float
            scale factor to use for the predictions (default corresponds to scale factor optimized for use with
            the activity index, if other activity measures are desired the scale factor can be modified or optimized.).
            Default: 0.30


        References
        ----------
        add reference

        """
        scale_factor = kwargs.get("scale_factor", 0.30)
        self.scale_factor: float = scale_factor
        """scale factor to use for the predictions (default corresponds to scale factor optimized for use with the
        activity index, if other activity measures are desired the scale factor can be modified or optimized.).
        Default: 0.30
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
              Default: 60

        Returns
        -------
        :obj:`~biopsykit.utils.datatype_helper.SleepWakeDataFrame`
            dataframe with sleep/wake predictions

        """
        index = None
        rescore_data: bool = kwargs.get("rescore_data", True)
        epoch_length: int = kwargs.get("epoch_length", 30)

        if isinstance(data, pd.DataFrame):
            index = data.index
            data = sanitize_input_1d(data)

        # ensure numpy
        # problem in my view: with np.convolve and "same" we dont make the "unsymmetric" convolution...
        # ( formula is with A-4 and A+2)
        # possible solution: np.valid and append 4 zeros in front and 2 zeros at the back or like implemented
        sf = np.array(self.scale_factor)
        kernel = sf * np.array(
            [
                0.0064,
                0.0074,
                0.0112,
                0.0112,
                0.0118,
                0.0118,
                0.0128,
                0.0188,
                0.0280,
                0.0664,
                0.0300,
                0.0112,
                0.100,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        scores = np.convolve(data, kernel, "same")

        scores[scores >= 1] = 99  # wake = 0
        scores[scores < 1] = 1  # sleep = 1
        scores[scores == 99] = 0  # wake = 0

        if rescore_data:
            scores = rescore(scores, epoch_length=epoch_length)

        if index is not None:
            scores = pd.DataFrame(scores, index=index, columns=["sleep_wake"])

        return _SleepWakeDataFrame(scores)
