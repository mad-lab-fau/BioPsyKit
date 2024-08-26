import warnings

import neurokit2 as nk
import numpy as np
import pandas as pd
from biopsykit.signals._base_extraction import BaseExtraction
from tpcp import make_action_safe


class QPeakExtractionNeurokitDwt(BaseExtraction):
    """algorithm to extract Q-wave peaks (= R-wave onset) from ECG signal using neurokit ecg_delineate function with
    discrete wavelet method.
    """

    @make_action_safe
    def extract(self, signal_clean: pd.DataFrame, heartbeats: pd.DataFrame, sampling_rate_hz: int):
        """Function which extracts Q-wave peaks from given ECG cleaned signal.

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
            saves resulting Q-peak locations (samples) in points_ attribute of super class (in the row of the heartbeat
            to which the respective Q-peak corresponds), index is heartbeat id,
            NaN when no Q-peak could be detected in that heartbeat
        """
        # result df
        q_peaks = pd.DataFrame(index=heartbeats.index, columns=["q_peak"])

        # used subsequently to store ids of heartbeats for which no AO or IVC could be detected
        heartbeats_no_q = []
        heartbeats_q_after_r = []

        # some neurokit functions (for example ecg_delineate()) don't work with r-peaks input as Series, so list instead
        r_peaks = list(heartbeats["r_peak_sample"])

        print(r_peaks)

        _, waves = nk.ecg_delineate(
            signal_clean, rpeaks=r_peaks, sampling_rate=sampling_rate_hz, method="dwt", show=False, show_type="peaks"
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
                    q_peaks.at[heartbeat_idx, "q_peak"] = np.NaN

                # valid Q-peak found
                else:
                    q_peaks.at[heartbeat_idx, "q_peak"] = q

        # inform user about missing Q-values
        if q_peaks.isna().sum()[0] > 0:
            nan_rows = q_peaks[q_peaks["q_peak"].isna()]
            nan_rows = nan_rows.drop(index=heartbeats_q_after_r)
            nan_rows = nan_rows.drop(index=heartbeats_no_q)
            warnings.warn(
                f"No Q-peak detected in {q_peaks.isna().sum()[0]} heartbeats (for heartbeats "
                f"{heartbeats_no_q} the neurokit algorithm was not able to detect a Q-peak; for heartbeats "
                f"{heartbeats_q_after_r} the detected Q is invalid because it occurs after R; for "
                f"{nan_rows.index.values} apparently none of the found Q-peaks were within these heartbeats)"
            )

        self.points_ = q_peaks
        return self
