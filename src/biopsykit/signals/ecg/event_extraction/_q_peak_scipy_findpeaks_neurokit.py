import warnings

import neurokit2 as nk
import numpy as np
import pandas as pd

from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS, CanHandleMissingEventsMixin
from biopsykit.signals._dtypes import assert_sample_columns_int
from biopsykit.signals.ecg.event_extraction._base_ecg_extraction import BaseEcgExtraction
from biopsykit.utils._datatype_validation_helper import _assert_has_columns, _assert_is_dtype
from biopsykit.utils.array_handling import sanitize_input_series
from biopsykit.utils.exceptions import EventExtractionError


class QPeakExtractionSciPyFindPeaksNeurokit(BaseEcgExtraction, CanHandleMissingEventsMixin):
    """Algorithm for Q-peak extraction using the scipy.find_peaks method implemented in NeuroKit2."""

    def __init__(self, handle_missing_events: HANDLE_MISSING_EVENTS = "warn"):
        """Initialize new QPeakExtractionMartinez2004Neurokit algorithm instance.

        Parameters
        ----------
        handle_missing_events : one of {"warn", "raise", "ignore"}, optional
            How to handle missing data in the input dataframes. Default: "warn"

        """
        super().__init__(handle_missing_events=handle_missing_events)

    # @make_action_safe
    def extract(
        self,
        *,
        ecg: pd.DataFrame,
        heartbeats: pd.DataFrame,
        sampling_rate_hz: int,
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
        q_peaks = pd.DataFrame(index=heartbeats.index, columns=["q_peak", "nan_reason"])

        # used subsequently to store ids of heartbeats for which no AO or IVC could be detected
        heartbeats_no_q = []
        heartbeats_q_after_r = []

        # some neurokit functions (for example ecg_delineate()) don't work with r-peaks input as Series, so list instead
        r_peaks = list(heartbeats["r_peak_sample"])

        _, waves = nk.ecg_delineate(ecg, rpeaks=r_peaks, sampling_rate=sampling_rate_hz, method="peaks", show=False)

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
                q_peaks.loc[heartbeats_no_q, "nan_reason"] = "no_q_peak"
                missing_str += (
                    f"- for heartbeats {heartbeats_no_q} the neurokit algorithm was not able to detect a Q-peak\n"
                )
            if len(heartbeats_q_after_r) > 0:
                q_peaks.loc[heartbeats_no_q, "nan_reason"] = "q_after_r_peak"
                missing_str += (
                    f"- for heartbeats {heartbeats_q_after_r} the detected Q is invalid "
                    f"because it occurs after the R-peak\n"
                )
            if len(nan_rows.index.values) > 0:
                q_peaks.loc[nan_rows.index, "nan_reason"] = "no_q_peak_within_heartbeats"
                missing_str += (
                    f"- for {nan_rows.index.to_numpy()} apparently none of the found Q-peaks "
                    f"were within these heartbeats"
                )

            if self.handle_missing_events == "warn":
                warnings.warn(missing_str)
            elif self.handle_missing_events == "raise":
                raise EventExtractionError(missing_str)

        _assert_is_dtype(q_peaks, pd.DataFrame)
        q_peaks.columns = ["q_peak_sample", "nan_reason"]
        _assert_has_columns(q_peaks, [["q_peak_sample", "nan_reason"]])
        q_peaks = q_peaks.astype({"q_peak_sample": "Int64", "nan_reason": "object"})
        assert_sample_columns_int(q_peaks)

        self.points_ = q_peaks
        return self
