from typing import Optional

import pandas as pd

from biopsykit.signals._base_extraction import BaseExtraction, EXTRACTION_HANDLING_BEHAVIOR
from biopsykit.signals.ecg.event_extraction._base_ecg_extraction import BaseEcgExtraction
from biopsykit.utils._datatype_validation_helper import _assert_is_dtype, _assert_has_columns


class RPeakExtraction(BaseEcgExtraction):
    """algorithm to extract Q-wave onset based on the detection of the R-peak."""

    # @make_action_safe
    def extract(
        self,
        ecg: pd.DataFrame,
        heartbeats: pd.DataFrame,
        sampling_rate_hz: int,
        *,
        handle_missing: Optional[EXTRACTION_HANDLING_BEHAVIOR] = "warn",
    ):
        """Function which extracts R-peaks from given ECG cleaned signal to use it as Q-wave onset estimate.

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
            saves resulting R-peak locations (samples) in points_ attribute of super class, index is heartbeat id
        """
        # get the r_peaks from the heartbeats Dataframe
        r_peaks = pd.DataFrame(index=heartbeats.index, columns=["R-peak"])
        r_peaks["R-peak"] = heartbeats["r_peak_sample"]

        r_peaks.columns = ["q_wave_onset_sample"]
        # points = super().match_points_heartbeats(self, points=points, heartbeats=heartbeats)

        # TODO handle missing

        _assert_is_dtype(r_peaks, pd.DataFrame)
        _assert_has_columns(r_peaks, [["q_wave_onset_sample"]])

        self.points_ = r_peaks
        return self
