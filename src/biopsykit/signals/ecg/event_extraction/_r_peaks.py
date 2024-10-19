from typing import Optional

import pandas as pd
from biopsykit.signals.ecg.event_extraction._base_ecg_extraction import BaseEcgExtraction
from biopsykit.utils._datatype_validation_helper import _assert_has_columns, _assert_is_dtype


class RPeakExtraction(BaseEcgExtraction):
    """algorithm to extract Q-wave onset based on the detection of the R-peak."""

    # @make_action_safe
    def extract(
        self,
        *,
        ecg: Optional[pd.DataFrame],  # noqa: ARG002
        heartbeats: pd.DataFrame,
        sampling_rate_hz: Optional[float],  # noqa: ARG002
    ):
        """Extract R-peaks from given ECG cleaned signal to use it as Q-wave onset estimate.

        The results are saved in the ``points_`` attribute of the super class.

        Parameters
        ----------
        ecg: :class:`~pandas.DataFrame`
            ECG signal. Not used in this function since R-peaks are extracted from the ``heartbeats`` DataFrame.
        heartbeats: :class:`~pandas.DataFrame`
            DataFrame containing one row per segmented heartbeat, each row contains start, end, and R-peak
            location (in samples from beginning of signal) of that heartbeat, index functions as id of heartbeat
        sampling_rate_hz: int
            Sampling rate of ECG signal in hz

        Returns
        -------
            self

        """
        # get the r_peaks from the heartbeats Dataframe
        r_peaks = pd.DataFrame(index=heartbeats.index, columns=["R-peak"])
        r_peaks["R-peak"] = heartbeats["r_peak_sample"]

        r_peaks.columns = ["q_wave_onset_sample"]
        # TODO handle missing

        _assert_is_dtype(r_peaks, pd.DataFrame)
        _assert_has_columns(r_peaks, [["q_wave_onset_sample"]])

        self.points_ = r_peaks
        return self
