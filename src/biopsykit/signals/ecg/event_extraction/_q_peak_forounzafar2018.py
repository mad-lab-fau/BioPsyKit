import numpy as np
import pandas as pd
from tpcp import Parameter

from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS, CanHandleMissingEventsMixin
from biopsykit.signals._dtypes import assert_sample_columns_int
from biopsykit.signals.ecg.event_extraction._base_ecg_extraction import BaseEcgExtraction
from biopsykit.utils._datatype_validation_helper import _assert_has_columns, _assert_is_dtype
from biopsykit.utils.array_handling import sanitize_input_series


class QPeakExtractionForouzanfar2018(BaseEcgExtraction, CanHandleMissingEventsMixin):
    """Algorithm by Forouzanfar et al. (2018) for Q-peak extraction."""

    scaling_factor: Parameter[float]

    def __init__(self, scaling_factor: float = 2000, handle_missing_events: HANDLE_MISSING_EVENTS = "warn"):
        """Initialize new QPeakExtractionVanLien algorithm instance.

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
        ecg: pd.DataFrame,
        heartbeats: pd.DataFrame,
        sampling_rate_hz: int,  # noqa: ARG002
    ):
        """Extract Q-peaks from given ECG cleaned signal.

        The results are saved in the ``points_`` attribute of the super class.

        Parameters
        ----------
        ecg: :class:`~pandas.DataFrame`
            ECG signal
        heartbeats: :class:`~pandas.DataFrame`
            DataFrame containing one row per segmented heartbeat, each row contains start, end, and R-peak
            location (in samples from beginning of signal) of that heartbeat, index functions as id of heartbeat
        sampling_rate_hz: int
            Sampling rate of ECG signal in hz

        Returns
        -------
            self

        Raises
        ------
        :exc:`~biopsykit.utils.exceptions.EventExtractionError`
            If missing data is found and ``handle_missing`` is set to "raise"

        """
        self._check_valid_missing_handling()
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
            ecg_before_r_peak = ecg[heartbeat_start:r_peak_sample].reset_index(drop=True)
            ecg_below = np.where(ecg_before_r_peak < threshold)[0]

            if len(ecg_below) == 0:
                q_peaks.loc[idx, "q_peak_sample"] = np.nan
                q_peaks.loc[idx, "nan_reason"] = "no_value_below_threshold"
                continue

            q_peak_sample = heartbeat_start + ecg_below[-1]
            q_peaks.loc[idx, "q_peak_sample"] = q_peak_sample

        _assert_is_dtype(q_peaks, pd.DataFrame)
        _assert_has_columns(q_peaks, [["q_peak_sample", "nan_reason"]])
        q_peaks = q_peaks.astype({"q_peak_sample": "Int64", "nan_reason": "object"})
        assert_sample_columns_int(q_peaks)

        self.points_ = q_peaks
        return self
