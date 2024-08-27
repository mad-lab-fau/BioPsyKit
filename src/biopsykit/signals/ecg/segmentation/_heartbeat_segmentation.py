from typing import Optional

import neurokit2 as nk
import numpy as np
import pandas as pd
from tpcp import Algorithm, Parameter, make_action_safe

__all__ = ["HeartBeatSegmentation"]


class HeartBeatSegmentation(Algorithm):
    """Finds R-peaks and segments ECG signal into heartbeats."""

    _action_methods = "extract"

    # input parameters
    variable_length: Parameter[bool]
    start_factor: Parameter[float]

    # result
    heartbeat_list_: pd.DataFrame

    def __init__(self, variable_length: Optional[bool] = True, start_factor: Optional[float] = 0.35):
        """Initialize new HeartBeatExtraction algorithm instance.

        Parameters
        ----------
        variable_length : bool
            ``True`` if extracted heartbeats should have variable length (depending on the current RR-interval) or
            ``False`` if extracted heartbeats should have fixed length (same length for all heartbeats, depending
            on the mean heartrate of the complete signal, 35% of mean heartrate in seconds before R-peak and 50%
            after r_peak, see :func:`neurokit2.ecg_segment` for details). For variable length heartbeats, the start of
            the next heartbeat follows directly after end of last (ends exclusive); For fixed length heartbeats,
            there might be spaces between heartbeat boarders, or they might overlap
        start_factor : float, optional
            only needed for variable_length heartbeats, factor between 0 and 1, which defines where the start boarder
            between heartbeats is set depending on the RR-interval to previous heartbeat
            For example, ``start_factor=0.35`` means that the beat start is set at 35% of current RR-distance before the
            R-peak of the beat
        """
        self.variable_length = variable_length
        self.start_factor = start_factor

    # @make_action_safe
    def extract(self, ecg_clean: pd.Series, sampling_rate_hz: int):
        """Segments ecg signal into heartbeats, extract start, end, r-peak of each heartbeat.

        fills df containing all heartbeats, one row corresponds to one heartbeat;
        for each heartbeat, df contains: start datetime, sample index of start/end, and sample index of r-peak;
        index of df can be used as heartbeat id

        Args:
            ecg_clean : containing cleaned ecg signal as pd series with datetime index
            sampling_rate_hz : containing sampling rate of ecg signal in hz as int
        Returns:
            self: fills heartbeat_list_
        """
        _, r_peaks = nk.ecg_peaks(ecg_clean, sampling_rate=sampling_rate_hz, method="neurokit")
        r_peaks = r_peaks["ECG_R_Peaks"]

        heartbeats = pd.DataFrame(
            index=np.arange(0, len(r_peaks)),
        )
        heartbeats = heartbeats.assign(r_peak_sample=r_peaks)

        # save RR-interval to successive heartbeat
        rr_interval_to_next_beat = np.abs(heartbeats["r_peak_sample"].diff(periods=-1))
        # rr_interval_to_next_beat.iloc[-1] = rr_interval_to_next_beat.iloc[-2]  # extrapolate last beat

        heartbeats = heartbeats.assign(rr_interval_sample=rr_interval_to_next_beat)

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

            if last_beat_end < len(ecg_clean):
                beat_ends.iloc[-1] = last_beat_end
            else:
                # drop the last beat if it is incomplete
                heartbeats = heartbeats.iloc[:-1]
                beat_ends = beat_ends.iloc[:-1]
                beat_starts = beat_starts.iloc[:-1]
            beat_ends = beat_ends.astype(int)

            # extract time of each beat's start
            beat_starts_time = ecg_clean.iloc[beat_starts].index
            heartbeats = heartbeats.assign(start_sample=beat_starts, end_sample=beat_ends, start_time=beat_starts_time)

        else:
            # split ecg signal into heartbeats with fixed length
            heartbeat_segments = nk.ecg_segment(ecg_clean, rpeaks=r_peaks, sampling_rate=sampling_rate_hz, show=False)

            heartbeat_segments_new = {int(k) - 1: v for k, v in heartbeat_segments.items()}
            heartbeat_segments_new = pd.concat(heartbeat_segments_new, names=["heartbeat_id"])

            heartbeat_segments_new = heartbeat_segments_new.groupby("heartbeat_id").agg(
                start_sample=("Index", "first"),
                end_sample=("Index", "last"),
                start_time=("Index", lambda s: ecg_clean.index[s.iloc[0]]),
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

        # ensure correct column order
        heartbeats = heartbeats[["start_time", "start_sample", "end_sample", "r_peak_sample"]]
        self.heartbeat_list_ = heartbeats
        return self
