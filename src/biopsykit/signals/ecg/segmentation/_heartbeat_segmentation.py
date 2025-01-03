from typing import Union

import neurokit2 as nk
import numpy as np
import pandas as pd
from tpcp import Parameter

from biopsykit.signals._dtypes import assert_sample_columns_int
from biopsykit.signals.ecg.segmentation._base_segmentation import BaseHeartbeatSegmentation

__all__ = ["HeartbeatSegmentationNeurokit"]

from biopsykit.utils._datatype_validation_helper import _assert_has_columns
from biopsykit.utils.array_handling import sanitize_input_dataframe_1d


class HeartbeatSegmentationNeurokit(BaseHeartbeatSegmentation):
    """Segments ECG signal into heartbeats based on Neurokit's delineate function."""

    _action_methods = "extract"

    # input parameters
    variable_length: Parameter[bool]
    start_factor: Parameter[float]
    r_peak_detection_method: Parameter[str]

    # result
    heartbeat_list_: pd.DataFrame

    def __init__(
        self,
        *,
        variable_length: bool = True,
        start_factor: float = 0.35,
        r_peak_detection_method: str = "neurokit",
    ):
        """Initialize new HeartBeatExtraction algorithm instance.

        Parameters
        ----------
        variable_length : bool, optional
            ``True`` if extracted heartbeats should have variable length (depending on the current RR-interval) or
            ``False`` if extracted heartbeats should have fixed length (same length for all heartbeats, depending
            on the mean heartrate of the complete signal, 35% of mean heartrate in seconds before R-peak and 50%
            after r_peak, see :func:`neurokit2.ecg_segment` for details).
            For variable length heartbeats, the start of the next heartbeat follows directly after end of last
            (ends exclusive); For fixed length heartbeats, there might be spaces between heartbeat borders, or they
            might overlap. Default: ``True``
        start_factor : float, optional
            only needed if ``variable_length=True``. If ``variable_length=True``, this parameter defines where the
            start border between heartbeats is set depending on the RR-interval to previous heartbeat.
            For example, ``start_factor=0.35`` means that the beat start is set at 35% of current RR-distance before the
            R-peak of the beat
        r_peak_detection_method : str, optional
            Method to detect R-peaks that is passed to :func:`neurokit2.ecg_peaks`. Default: "neurokit"

        """
        self.variable_length = variable_length
        self.start_factor = start_factor
        self.r_peak_detection_method = r_peak_detection_method

    # @make_action_safe
    def extract(
        self,
        *,
        ecg: Union[pd.Series, pd.DataFrame],
        sampling_rate_hz: int,
    ):
        """Segments ecg signal into heartbeats, extract start, end, r-peak of each heartbeat.

        The function uses R-peak detection to segment the ECG signal into heartbeats. The start of each heartbeat is
        determined based on the R-peak and the current RR-interval.

        The results are saved in the ``heartbeat_list_`` attribute.

        Parameters
        ----------
        ecg : :class:`~pandas.Series` or :class:`~pandas.DataFrame`
            ECG signal
        sampling_rate_hz : int
            Sampling rate of ECG signal in hz

        Returns
        -------
            self

        """
        ecg = sanitize_input_dataframe_1d(ecg, column="ECG")
        if ecg.empty:
            raise ValueError("Input data is empty!")
        _assert_has_columns(ecg, [["ECG"]])

        _, r_peaks = nk.ecg_peaks(ecg, sampling_rate=sampling_rate_hz, method=self.r_peak_detection_method)
        r_peaks = r_peaks["ECG_R_Peaks"]

        heartbeats = pd.DataFrame(
            index=np.arange(0, len(r_peaks)),
        )
        heartbeats = heartbeats.assign(r_peak_sample=r_peaks)

        # save RR-interval to successive heartbeat
        heartbeats = heartbeats.assign(rr_interval_sample=np.abs(heartbeats["r_peak_sample"].diff(periods=-1)))

        if self.variable_length:
            # split ecg signal into heartbeats with varying length
            rr_interval_samples = heartbeats["r_peak_sample"].diff()

            # calculate start of each heartbeat based on corresponding R-peak and current RR-interval
            beat_starts = heartbeats["r_peak_sample"] - self.start_factor * rr_interval_samples

            # extrapolate first beats start based on RR-interval of next beat
            first_beat_start = heartbeats["r_peak_sample"].iloc[0] - self.start_factor * rr_interval_samples.iloc[1]
            if first_beat_start >= 0:
                beat_starts.iloc[0] = first_beat_start
            else:
                beat_starts = beat_starts.iloc[1:].reset_index(drop=True)  # drop row if heartbeat is incomplete
                heartbeats = heartbeats.iloc[1:].reset_index(drop=True)
            beat_starts = round(beat_starts).astype(int)

            # calculate beat ends (last beat ends 1 sample before next starts, end is exclusive)
            beat_ends = beat_starts.shift(-1)  # end is exclusive

            # extrapolate last beats end based on RR-interval of previous beat
            last_beat_end = round(
                heartbeats["r_peak_sample"].iloc[-1] + (1 - self.start_factor) * rr_interval_samples.iloc[-1]
            )

            if last_beat_end < len(ecg):
                beat_ends.iloc[-1] = last_beat_end
            else:
                # drop the last beat if it is incomplete
                heartbeats = heartbeats.iloc[:-1]
                beat_ends = beat_ends.iloc[:-1]
                beat_starts = beat_starts.iloc[:-1]
            beat_ends = beat_ends.astype(int)

            # extract time of each beat's start
            beat_starts_time = ecg.iloc[beat_starts].index
            heartbeats = heartbeats.assign(start_sample=beat_starts, end_sample=beat_ends, start_time=beat_starts_time)

        else:
            # split ecg signal into heartbeats with fixed length
            heartbeat_segments = nk.ecg_segment(ecg, rpeaks=r_peaks, sampling_rate=sampling_rate_hz, show=False)

            heartbeat_segments_new = {int(k) - 1: v for k, v in heartbeat_segments.items()}
            heartbeat_segments_new = pd.concat(heartbeat_segments_new, names=["heartbeat_id"])

            heartbeat_segments_new = heartbeat_segments_new.groupby("heartbeat_id").agg(
                start_sample=("Index", "first"),
                end_sample=("Index", "last"),
                start_time=("Index", lambda s: ecg.index[s.iloc[0]]),
            )
            # fill the empty columns of heartbeats with the start, end, and r-peak of heartbeat_segments_new
            heartbeats = heartbeats.join(heartbeat_segments_new)

        # check if R-peak occurs between corresponding start and end
        check = heartbeats.apply(lambda x: x["start_sample"] < x["r_peak_sample"] < x["end_sample"], axis=1)
        if len(check.loc[~check]) > 0:
            raise ValueError(
                f"Start/end/R-peak position of heartbeat {list(check.loc[check is False].index)} could be incorrect!"
            )

        # ensure that index is Int64Index (not RangeIndex) because some neurokit functions won't work  with RangeIndex
        heartbeats.index = list(heartbeats.index)
        heartbeats.index.name = "heartbeat_id"

        heartbeats = heartbeats.assign(rr_interval_ms=heartbeats["rr_interval_sample"] / sampling_rate_hz * 1000)

        # ensure correct column order
        heartbeats = heartbeats[
            ["start_time", "start_sample", "end_sample", "r_peak_sample", "rr_interval_sample", "rr_interval_ms"]
        ]

        heartbeats = heartbeats.convert_dtypes(infer_objects=True)
        # assert that columns with "_sample" in the end are of type int
        assert_sample_columns_int(heartbeats)

        self.heartbeat_list_ = heartbeats
        return self
