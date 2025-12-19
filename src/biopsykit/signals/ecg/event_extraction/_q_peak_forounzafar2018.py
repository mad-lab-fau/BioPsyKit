import numpy as np
import pandas as pd
from tpcp import Parameter

from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS, CanHandleMissingEventsMixin
from biopsykit.signals.ecg.event_extraction._base_ecg_extraction import BaseEcgExtractionWithHeartbeats
from biopsykit.utils.array_handling import sanitize_input_series

__all__ = ["QPeakExtractionForouzanfar2018"]

from biopsykit.utils.dtypes import (
    EcgRawDataFrame,
    HeartbeatSegmentationDataFrame,
    is_ecg_raw_dataframe,
    is_heartbeat_segmentation_dataframe,
    is_q_peak_dataframe,
)


class QPeakExtractionForouzanfar2018(BaseEcgExtractionWithHeartbeats, CanHandleMissingEventsMixin):
    r"""Q-peak extraction algorithm by Forouzanfar et al. (2018).

    This algorithm detects the Q-peak of an ECG signal based on the last sample before the R-peak that is below a
    certain threshold (:math:`-1.2 \cdot R_peak/f_s`), where R_peak is the amplitude of the R-peak and f_s is the
    sampling frequency of the ECG signal.

    In this implementation, a *scaling_factor* is used instead of *f_s* since the sampling rates of the datasets
    can differ from the original publication (2000 Hz). Thus, the scaling_factor is set to the sampling rate of the
    original publication (2000 Hz) by default.

    For more information on the algorithm, see [For18]_.


    References
    ----------
    .. [For18] Forouzanfar, M., Baker, F. C., De Zambotti, M., McCall, C., Giovangrandi, L., & Kovacs, G. T. A. (2018).
        Toward a better noninvasive assessment of preejection period: A novel automatic algorithm for B-point detection
        and correction on thoracic impedance cardiogram. Psychophysiology, 55(8), e13072.
        https://doi.org/10.1111/psyp.13072


    """

    scaling_factor: Parameter[float]

    def __init__(self, scaling_factor: float = 2000, handle_missing_events: HANDLE_MISSING_EVENTS = "warn"):
        """Initialize new ``QPeakExtractionForouzanfar2018`` algorithm instance.

        Parameters
        ----------
        scaling_factor : float, optional
            Scaling factor for the threshold used to detect the Q-peak. Default: 2000
        handle_missing_events : one of {"warn", "raise", "ignore"}, optional
            How to handle missing data in the input dataframes. Default: "warn"
        """
        super().__init__(handle_missing_events=handle_missing_events)
        self.scaling_factor = scaling_factor

    # @make_action_safe
    def extract(
        self,
        *,
        ecg: EcgRawDataFrame,
        heartbeats: HeartbeatSegmentationDataFrame,
        sampling_rate_hz: float,  # noqa: ARG002
    ):
        """Extract Q-peaks from given ECG signal.

        The results are saved in the ``points_`` attribute of the super class.

        Parameters
        ----------
        ecg: :class:`~pandas.DataFrame`
            ECG signal
        heartbeats: :class:`~pandas.DataFrame`
            DataFrame containing one row per segmented heartbeat, each row contains start, end, and R-peak
            location (in samples from beginning of signal) of that heartbeat, index functions as id of heartbeat
        sampling_rate_hz: float
            Sampling rate of ECG signal in hz

        Returns
        -------
            self

        Raises
        ------
        :exc:`~biopsykit.utils.exceptions.EventExtractionError`
            If the event extraction fails and ``handle_missing`` is set to "raise"

        """
        self._check_valid_missing_handling()
        is_ecg_raw_dataframe(ecg)
        is_heartbeat_segmentation_dataframe(heartbeats)
        ecg = sanitize_input_series(ecg, name="ecg")
        ecg = ecg.squeeze()

        # result df
        q_peaks = pd.DataFrame(index=heartbeats.index, columns=["q_peak_sample", "nan_reason"])

        # search Q-peak for each heartbeat of the given signal
        for idx, data in heartbeats.iterrows():
            heartbeat_start = data["start_sample"]
            r_peak_sample = data["r_peak_sample"]

            # set an individual threshold for detecting the Q-peaks based on the R-peak
            threshold = (-1.2 * ecg.iloc[r_peak_sample]) / self.scaling_factor

            # search for the Q-peak as the last sample before the R-peak that is below the threshold
            ecg_before_r_peak = ecg.iloc[heartbeat_start:r_peak_sample].reset_index(drop=True)
            ecg_below = np.where(ecg_before_r_peak < threshold)[0]

            if len(ecg_below) == 0:
                q_peaks.loc[idx, "q_peak_sample"] = np.nan
                q_peaks.loc[idx, "nan_reason"] = "no_value_below_threshold"
                continue

            q_peak_sample = heartbeat_start + ecg_below[-1]
            q_peaks.loc[idx, "q_peak_sample"] = q_peak_sample

        q_peaks = q_peaks.astype({"q_peak_sample": "Int64", "nan_reason": "object"})
        is_q_peak_dataframe(q_peaks)

        self.points_ = q_peaks
        return self
