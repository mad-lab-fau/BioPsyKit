from typing import Optional

import pandas as pd
from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS
from biopsykit.signals.icg.event_extraction._base_c_point_extraction import BaseCPointExtraction

__all__ = ["CPointExtractionKoka2022"]


class CPointExtractionKoka2022(BaseCPointExtraction):
    """Extract C-points from ICG derivative signal using the method proposed by Koka et al. (2022).

    This method is based on the ECG R-peak detection algorithm by Koka et al. (2022).

    References
    ----------
    Koka, T., & Muma, M. (2022). Fast and Sample Accurate R-Peak Detection for Noisy ECG Using Visibility Graphs.
    2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), 121-126.
    https://doi.org/10.1109/EMBC48229.2022.9871266

    """

    # @make_action_safe
    def extract(
        self,
        *,
        icg: pd.Series,
        heartbeats: pd.DataFrame,
        sampling_rate_hz: int,
        handle_missing: Optional[HANDLE_MISSING_EVENTS] = "warn",
    ):
        """Extract C-points from given cleaned ICG derivative signal using :func:`~neurokit2.ecg_peaks` with
        the method "koka2022".

        The results are saved in the 'points_' attribute of the class instance.

        Parameters
        ----------
        icg : :class:`~pandas.DataFrame`
            cleaned ICG derivative signal
        heartbeats : :class:`~pandas.DataFrame`
            Dataframe containing one row per segmented heartbeat, each row contains start, end, and R-peak location
            (in samples from beginning of signal) of that heartbeat, index functions as id of heartbeat
        sampling_rate_hz : int
            sampling rate of ICG derivative signal in hz
        handle_missing : one of {"warn", "raise", "ignore"}, optional
            How to handle missing data in the input dataframes. Not used in this function.

        Returns
        -------
            self
        """  # noqa: D205
        raise NotImplementedError("This function is not implemented yet.")
        # # result df
        # c_points = pd.DataFrame(index=heartbeats.index, columns=["c_point"])
        #
        # signal_clean.columns = ["ECG_Clean"]
        #
        # # search C-point for each heartbeat of the given signal
        # for idx, data in heartbeats.iterrows():
        #     # slice signal for current heartbeat
        #     heartbeat_start = data["start_sample"]
        #     heartbeat_end = data["end_sample"]
        #     heartbeat_icg_der = signal_clean.iloc[heartbeat_start:heartbeat_end]
        #
        #     c_point = nk.ecg_peaks(heartbeat_icg_der, sampling_rate_hz, method="koka2022")
        #
        #     c_points["c_point"].iloc[idx] = c_point
        #
        # self.points_ = c_points
        # return self
