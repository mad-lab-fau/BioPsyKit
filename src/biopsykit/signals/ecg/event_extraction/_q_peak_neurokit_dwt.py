import warnings
from typing import Optional

import neurokit2 as nk
import numpy as np
import pandas as pd
from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS
from biopsykit.signals.ecg.event_extraction._base_ecg_extraction import BaseEcgExtraction
from biopsykit.utils._datatype_validation_helper import _assert_has_columns, _assert_is_dtype
from biopsykit.utils.exceptions import EventExtractionError


class QPeakExtractionNeurokitDwt(BaseEcgExtraction):
    """Q-wave peaks extraction using :func:`~neurokit2.ecg_delineate` function with discrete wavelet method."""

    # @make_action_safe
    def extract(
        self,
        ecg: pd.DataFrame,
        heartbeats: pd.DataFrame,
        sampling_rate_hz: int,
        *,
        handle_missing: Optional[HANDLE_MISSING_EVENTS] = "warn",
    ):
        """Extract Q-wave peaks from given ECG cleaned signal.

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
        handle_missing : one of {"warn", "raise", "ignore"}, optional
            How to handle missing data in the input dataframes. If "warn", a warning is raised if missing data is found.
            If "raise", an exception is raised if missing data is found. If "ignore", missing data is ignored.
            Default: "warn"

        Returns
        -------
            self

        Raises
        ------
        :exc:`~biopsykit.utils.exceptions.EventExtractionError`
            If missing data is found and ``handle_missing`` is set to "raise"

        """
        # result df
        q_peaks = pd.DataFrame(index=heartbeats.index, columns=["q_peak"])

        # used subsequently to store ids of heartbeats for which no AO or IVC could be detected
        heartbeats_no_q = []
        heartbeats_q_after_r = []

        # some neurokit functions (for example ecg_delineate()) don't work with r-peaks input as Series, so list instead
        r_peaks = list(heartbeats["r_peak_sample"])

        ecg = ecg.squeeze()
        _, waves = nk.ecg_delineate(
            ecg, rpeaks=r_peaks, sampling_rate=sampling_rate_hz, method="dwt", show=False, show_type="peaks"
        )  # show can also be set to False

        extracted_q_peaks = waves["ECG_Q_Peaks"]

        # find heartbeat to which Q-peak belongs and save Q-peak position in corresponding row
        for idx, q in enumerate(extracted_q_peaks):
            # for some heartbeats, no Q can be detected, will be NaN in resulting df
            if np.isnan(q):
                heartbeats_no_q.append(idx)
            else:
                heartbeat_idx = heartbeats.loc[(heartbeats["start_sample"] < q) & (q < heartbeats["end_sample"])].index[
                    0
                ]

                # Q occurs after R, which is not valid
                if heartbeats["r_peak_sample"].loc[heartbeat_idx].item() < q:
                    heartbeats_q_after_r.append(heartbeat_idx)
                    q_peaks.loc[heartbeat_idx, "q_peak"] = np.NaN
                # valid Q-peak found
                else:
                    q_peaks.loc[heartbeat_idx, "q_peak"] = q

        # inform user about missing Q-values
        if q_peaks.isna().sum()[0] > 0:
            nan_rows = q_peaks[q_peaks["q_peak"].isna()]
            nan_rows = nan_rows.drop(index=heartbeats_q_after_r)
            nan_rows = nan_rows.drop(index=heartbeats_no_q)

            missing_str = f"No Q-peak detected in {q_peaks.isna().sum()[0]} heartbeats:\n"
            if len(heartbeats_no_q) > 0:
                missing_str += (
                    f"- for heartbeats {heartbeats_no_q} the neurokit algorithm was not able to detect a Q-peak\n"
                )
            if len(heartbeats_q_after_r) > 0:
                missing_str += (
                    f"- for heartbeats {heartbeats_q_after_r} the detected Q is invalid "
                    f"because it occurs after the R-peak\n"
                )
            if len(nan_rows.index.values) > 0:
                missing_str += (
                    f"- for {nan_rows.index.to_numpy()} apparently none of the found Q-peaks "
                    f"were within these heartbeats"
                )

            if handle_missing == "warn":
                warnings.warn(missing_str)
            elif handle_missing == "raise":
                raise EventExtractionError(missing_str)

        q_peaks.columns = ["q_wave_onset_sample"]

        _assert_is_dtype(q_peaks, pd.DataFrame)
        _assert_has_columns(q_peaks, [["q_wave_onset_sample"]])

        self.points_ = q_peaks
        return self
