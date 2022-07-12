"""Sleep/Wake detection using the *Cole/Kripke Algorithm*."""
import numpy as np
import pandas as pd

from biopsykit.sleep.sleep_wake_detection.algorithms._base import _SleepWakeBase
from biopsykit.sleep.sleep_wake_detection.utils import rescore
from biopsykit.utils._types import arr_t
from biopsykit.utils.array_handling import sanitize_input_1d
from biopsykit.utils.datatype_helper import SleepWakeDataFrame, _SleepWakeDataFrame


class ColeKripke(_SleepWakeBase):
    """Class representing the *Cole/Kripke Algorithm* for sleep/wake detection."""

    def __init__(self, **kwargs):  # pylint: disable=useless-super-delegation
        """Class representing the *Cole/Kripke Algorithm* for sleep/wake detection.

        The *Cole/Kripke Algorithm* runs sleep wake detection on epoch level activity data. Epochs are 1 minute long and
        activity is represented by an activity index which comes from Actigraph data or from raw acceleration data
        converted into activity index data.

        Parameters
        ----------
        epoch_length : int
            epoch length in seconds. Epoch lengths are usually 10, 30, or 60 seconds.
        scale_factor : float
            scale factor to use for the predictions (default corresponds to scale factor optimized for use with
            the activity index, if other activity measures are desired the scale factor can be modified or optimized.)
            The recommended range for the scale factor is between 0.1 and 0.25 depending on the sensitivity to activity
            desired, and possibly the population being observed. According to the paper by Cole and Kripke,
            the scale factor depends on the epoch length. See the paper for more details.

        References
        ----------
        Cole, R. J., Kripke, D. F., Gruen, W., Mullaney, D. J., & Gillin, J. C. (1992). Automatic Sleep/Wake
        Identification From Wrist Activity. *Sleep*, 15(5), 461–469. https://doi.org/10.1093/sleep/15.5.461

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
        """Apply sleep/wake prediction on activity index values.

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

        if isinstance(data, pd.DataFrame):
            index = data.index
            data = sanitize_input_1d(data)

        # ensure numpy
        # problem in my view: with np.convolve and "same" we don't make the "unsymmetrical" convolution...
        # ( formula is with A-4 and A+2)
        # possible solution: np.valid and append 4 zeros in front and 2 zeros at the back or like implemented
        scores = np.convolve(data, self._get_kernel(), "same")

        scores[scores >= 1] = 99  # wake = 0
        scores[scores < 1] = 1  # sleep = 1
        scores[scores == 99] = 0  # wake = 0       #changed to 1 according to paper

        if rescore_data:
            scores = rescore(scores, epoch_length=self.epoch_length)

        if index is not None:
            scores = pd.DataFrame(scores, index=index, columns=["sleep_wake"])

        return _SleepWakeDataFrame(scores)

    def _get_kernel(self):
        """Get convolution kernel for a given epoch length."""
        scale_map = {
            10: 0.00001,
            30: 0.0001,
            60: 0.001,
        }
        coeff_map = {
            10: [550, 378, 413, 699, 1736, 287, 309, 0, 0],
            30: [50, 30, 14, 28, 121, 8, 50, 0, 0],
            60: [106, 54, 58, 76, 230, 74, 67, 0, 0],
        }
        sf = np.array(scale_map[self.epoch_length])
        return sf * np.array(coeff_map[self.epoch_length])
