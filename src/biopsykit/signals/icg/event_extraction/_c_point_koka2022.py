from typing import Optional

import neurokit2 as nk
import pandas as pd
from biopsykit.signals._base_extraction import BaseExtraction, EXTRACTION_HANDLING_BEHAVIOR
from tpcp import make_action_safe


class CPointExtractionKoka2022(BaseExtraction):
    """algorithm to extract C-points from ICG derivative signal using neurokit2s ecg_peaks() with the method koka2022."""

    # @make_action_safe
    def extract(
        self,
        signal_clean: pd.DataFrame,
        heartbeats: pd.DataFrame,
        sampling_rate_hz: int,
        *,
        handle_missing: Optional[EXTRACTION_HANDLING_BEHAVIOR] = "warn",
    ):
        """Function which extracts C-points (max of most prominent peak) from given cleaned ICG derivative signal.

        Args:
            signal_clean:
                cleaned ICG derivative signal
            heartbeats:
                pd.DataFrame containing one row per segmented heartbeat, each row contains start, end, and R-peak
                location (in samples from beginning of signal) of that heartbeat, index functions as id of heartbeat
            sampling_rate_hz:
                sampling rate of ICG derivative signal in hz

        Returns
        -------
            saves resulting C-point positions in points_, index is heartbeat id
        """
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
