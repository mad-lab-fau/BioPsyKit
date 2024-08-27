from typing import Optional

import pandas as pd
from biopsykit.signals._base_extraction import BaseExtraction, EXTRACTION_HANDLING_BEHAVIOR
from tpcp import Parameter, make_action_safe

from biopsykit.signals.ecg.event_extraction._base_ecg_extraction import BaseEcgExtraction
from biopsykit.utils._datatype_validation_helper import _assert_is_dtype, _assert_has_columns


class QWaveOnsetExtractionVanLien(BaseEcgExtraction):
    """algorithm to extract Q-wave onset based on the detection of the R-peak
    and the subtraction of a fixed time interval.
    """

    # parameters
    time_interval: Parameter[int]

    def __init__(self, time_interval: Optional[int] = 40):
        """Initialize new QWaveOnsetExtractionVanLien algorithm instance.

        Params:
        time_interval : int
            Specify the constant time interval which will be subtracted from the R-peak for Q-wave onset estimation
        """
        self.time_interval = time_interval

    # @make_action_safe
    def extract(
        self,
        ecg: pd.DataFrame,
        heartbeats: pd.DataFrame,
        sampling_rate_hz: int,
        *,
        handle_missing: Optional[EXTRACTION_HANDLING_BEHAVIOR] = "warn",
    ):
        """Function which extracts Q-wave onset (start of ventricular depolarization) from given ECG cleaned signal.

        Args:
            signal_clean:
                cleaned ECG signal
            heartbeats:
                pd.DataFrame containing one row per segmented heartbeat, each row contains start, end, and R-peak
                location (in samples from beginning of signal) of that heartbeat, index functions as id of heartbeat
            sampling_rate_hz:
                sampling rate of ECG signal in hz

        Returns
        -------
            saves resulting Q-wave-onset locations (samples) in points_ attribute of super class, index is heartbeat id
        """
        # convert the fixed time_interval from milliseconds into samples
        time_interval_in_samples = (self.time_interval / 1000) * sampling_rate_hz  # 40 ms = 0.04 s

        # get the r_peaks from the heartbeats Dataframe
        r_peaks = heartbeats[["r_peak_sample"]]

        # subtract the fixed time_interval from the r_peak samples to estimate the q_wave_onset
        q_wave_onset = r_peaks - time_interval_in_samples

        q_wave_onset.columns = ["q_wave_onset_sample"]

        # TODO handle missing

        _assert_is_dtype(q_wave_onset, pd.DataFrame)
        _assert_has_columns(q_wave_onset, [["q_wave_onset_sample"]])

        self.points_ = q_wave_onset
        return self
