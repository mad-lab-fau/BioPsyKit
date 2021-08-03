"""Sleep/Wake detection using the *Cole/Kripke Algorithm*."""
import numpy as np
import pandas as pd
from biopsykit.sleep.sleep_wake_detection.algorithms._base import _SleepWakeBase
from biopsykit.utils._types import arr_t
from biopsykit.utils.array_handling import sanitize_input_1d
from biopsykit.utils.datatype_helper import SleepWakeDataFrame, _SleepWakeDataFrame


class ColeKripke(_SleepWakeBase):
    """Class representing the *Cole/Kripke Algorithm* for sleep/wake detection based on activity counts.

    The *Cole/Kripke Algorithm* runs sleep wake detection on epoch level activity data. Epochs are 1 minute long and
    activity is represented by an activity index which comes from Actigraph data or from raw acceleration data
    converted into activity index data.

    References
    ----------
    Cole, R. J., Kripke, D. F., Gruen, W., Mullaney, D. J., & Gillin, J. C. (1992). Automatic Sleep/Wake
    Identification From Wrist Activity. *Sleep*, 15(5), 461–469. https://doi.org/10.1093/sleep/15.5.461

    """

    scale_factor: float

    def __init__(self, **kwargs):
        """Initialize a new ``ColeKripke`` instance.

        Parameters
        ----------
        scale_factor : float
            scale factor to use for the predictions (default corresponds to scale factor optimized for use with
            the activity index, if other activity measures are desired the scale factor can be modified or optimized.)
            The recommended range for the scale factor is between 0.1 and 0.25 depending on the sensitivity to activity
            desired, and possibly the population being observed.

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
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0

        # rescore the original predictions
        scores = self._rescore(scores)

        scores = pd.DataFrame(scores, index=index, columns=["sleep_wake"])
        return _SleepWakeDataFrame(scores)

    @staticmethod
    def _rescore(predictions: np.array) -> np.array:  # noqa: C901
        """Apply Webster's rescoring rules.

        Parameters
        ----------
        predictions : array_like
            sleep/wake predictions

        Returns
        -------
        array_like
            rescored sleep/wake predictions

        """
        rescored = predictions.copy()

        # rules a through c
        wake_bin = 0
        for t in range(len(rescored)):  # pylint:disable=consider-using-enumerate
            if rescored[t] == 1:
                wake_bin += 1
            else:
                if wake_bin >= 15:
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
        for t in range(1, len(rescored) - 1):  # pylint:disable=consider-using-enumerate
            if rescored[t] == 1 and rescored[t - 1] == 0 and rescored[t + 1] == 0:
                rescored[t] = 0

        return rescored
