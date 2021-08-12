"""Sleep/Wake detection using the *Cole/Kripke Algorithm*."""
import numpy as np
import pandas as pd

from biopsykit.sleep.sleep_wake_detection.algorithms._base import _SleepWakeBase
from biopsykit.sleep.sleep_wake_detection.utils import rescore
from biopsykit.utils._types import arr_t
from biopsykit.utils.array_handling import sanitize_input_1d
from biopsykit.utils.datatype_helper import SleepWakeDataFrame, _SleepWakeDataFrame


class ColeKripke(_SleepWakeBase):
    """Class representing the *Cole/Kripke Algorithm* for sleep/wake detection based on activity counts."""

    def __init__(self, **kwargs):
        """Class representing the *Cole/Kripke Algorithm* for sleep/wake detection based on activity counts.

        The *Cole/Kripke Algorithm* runs sleep wake detection on epoch level activity data. Epochs are 1 minute long and
        activity is represented by an activity index which comes from Actigraph data or from raw acceleration data
        converted into activity index data.

        Parameters
        ----------
        scale_factor : float
            scale factor to use for the predictions (default corresponds to scale factor optimized for use with
            the activity index, if other activity measures are desired the scale factor can be modified or optimized.)
            The recommended range for the scale factor is between 0.1 and 0.25 depending on the sensitivity to activity
            desired, and possibly the population being observed.

        References
        ----------
        Cole, R. J., Kripke, D. F., Gruen, W., Mullaney, D. J., & Gillin, J. C. (1992). Automatic Sleep/Wake
        Identification From Wrist Activity. *Sleep*, 15(5), 461–469. https://doi.org/10.1093/sleep/15.5.461

        """
        self.scale_factor: float = kwargs.pop("scale_factor", None)
        """Scale factor to use for the predictions (default corresponds to scale factor optimized for use with the
        activity index, if other activity measures are desired the scale factor can be modified or optimized).
        The recommended range for the scale factor is between 0.1 and 0.25 depending on the sensitivity to activity
        desired, and possibly the population being observed.
        """

        if self.scale_factor is None:
            self.scale_factor = 0.193125
        super().__init__(**kwargs)

    def fit(self, data: arr_t, **kwargs) -> arr_t:
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
        """Apply sleep/wake prediction on activity index values.

        Parameters
        ----------
        data : array_like
            array with activity index values

        Returns
        -------
        :obj:`~biopsykit.utils.datatype_helper.SleepWakeDataFrame`
            dataframe with sleep/wake predictions

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
        scores = rescore(scores)

        scores = pd.DataFrame(scores, index=index, columns=["sleep_wake"])
        return _SleepWakeDataFrame(scores)
