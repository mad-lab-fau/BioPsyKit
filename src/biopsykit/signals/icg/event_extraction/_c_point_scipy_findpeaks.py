import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import signal
from tpcp import Parameter

from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS, CanHandleMissingEventsMixin
from biopsykit.signals._dtypes import assert_sample_columns_int
from biopsykit.signals.icg.event_extraction._base_c_point_extraction import BaseCPointExtraction
from biopsykit.utils._datatype_validation_helper import _assert_has_columns, _assert_is_dtype
from biopsykit.utils.array_handling import sanitize_input_dataframe_1d
from biopsykit.utils.exceptions import EventExtractionError

__all__ = ["CPointExtractionScipyFindPeaks"]


class CPointExtractionScipyFindPeaks(BaseCPointExtraction, CanHandleMissingEventsMixin):
    """Extract C-points from ICG derivative signal using scipy's find_peaks function."""

    # input parameters
    window_c_correction: Parameter[int]

    def __init__(
        self,
        window_c_correction: Optional[int] = 3,
        handle_missing_events: HANDLE_MISSING_EVENTS = "warn",
    ):
        """Initialize the C-point extraction algorithm.

        Parameters
        ----------
        window_c_correction : int, optional
            how many preceding heartbeats are taken into account for C-point correction (using mean R-C-distance)
        handle_missing_events : one of {"warn", "raise", "ignore"}, optional
            How to handle missing C-points (default: "warn").
            * "warn" : issue a warning and set C-point to NaN
            * "raise" : raise an :class:`~biopsykit.utils.exceptions.EventExtractionError`
            * "ignore" : ignore missing C-points

        """
        super().__init__(handle_missing_events=handle_missing_events)
        self.window_c_correction = window_c_correction

    # @make_action_safe
    def extract(
        self,
        *,
        icg: Union[pd.Series, pd.DataFrame],
        heartbeats: pd.DataFrame,
        sampling_rate_hz: Optional[float],  # noqa: ARG002
    ):
        """Extract C-points from given cleaned ICG derivative signal using :func:`~scipy.signal.find_peaks`.

        The C-point is detected as the maximum of the most prominent peak in the ICG derivative signal within each
        segmented heartbeat.

        The resulting C-points are saved in the 'points_' attribute of the class instance.

        Parameters
        ----------
        icg : :class:`~pandas.Series`
            cleaned ICG derivative signal
        heartbeats : :class:`~pandas.DataFrame`
            Dataframe containing one row per segmented heartbeat, each row contains start, end, and R-peak.
            Result from :class:`~biopsykit.signals.ecg.segmentation.HeartbeatSegmentation`.
        sampling_rate_hz : int
            Sampling rate of ICG derivative signal in Hz. Not used in this function.

        Returns
        -------
        self

        """
        self._check_valid_missing_handling()
        icg = sanitize_input_dataframe_1d(icg, column="icg_der")

        # result df
        c_points = pd.DataFrame(index=heartbeats.index, columns=["c_point_sample", "nan_reason"])

        # distance of R-peak to C-point, averaged over as many preceding heartbeats as window_c_correction specifies
        # R-C-distances are positive when C-point occurs after R-Peak (which is the physiologically correct order)
        mean_prev_r_c_distance = np.NaN

        # saves R-C-distances of previous heartbeats
        prev_r_c_distances = []

        # used subsequently to store heartbeats for which no C-point could be detected
        heartbeats_no_c = []

        # search C-point for each heartbeat of the given signal
        for idx, data in heartbeats.iterrows():
            # slice signal for current heartbeat
            heartbeat_start = data["start_sample"]
            heartbeat_end = data["end_sample"]

            heartbeat_icg_der = icg.iloc[heartbeat_start:heartbeat_end].squeeze()

            # calculate R-peak position relative to start of current heartbeat
            heartbeat_r_peak = data["r_peak_sample"] - heartbeat_start

            # detect possible C-point candidates, prominence=1 gives reasonable Cs
            # (prominence=2 results in a considerable amount of heartbeats with no C)
            # (prominence=1 might detect more than one C in one heartbeat, but that will be corrected subsequently)
            heartbeat_c_candidates = signal.find_peaks(heartbeat_icg_der, prominence=1)[0]

            if len(heartbeat_c_candidates) < 1:
                heartbeats_no_c.append(idx)
                c_points.loc[idx, "c_point_sample"] = np.NaN
                continue

            # calculates distance of R-peak to all C-candidates in samples, positive when C occurs after R
            r_c_distance = heartbeat_c_candidates - heartbeat_r_peak

            if len(heartbeat_c_candidates) == 1:
                selected_c = heartbeat_c_candidates[0]  # convert to int (instead of array)
                r_c_distance = r_c_distance[0]

                # C-point before R-peak is invalid
                if r_c_distance < 0:
                    heartbeats_no_c.append(idx)
                    c_points.loc[idx, "c_point_sample"] = np.NaN
                    continue
            else:
                # take averaged R-C-distance over the 'window_c_correction' (default: 3) preceding heartbeats
                # calculate the absolute difference of R-C-distances for all C-candidates to this mean
                # (to check which of the C-candidates are most probably the wrongly detected Cs)
                distance_diff = np.abs(r_c_distance - mean_prev_r_c_distance)

                # choose the C-candidate with the smallest absolute difference in R-C-distance
                # (the one, where R-C-distance changed the least compared to previous heartbeats)
                c_idx = np.argmin(distance_diff)
                selected_c = heartbeat_c_candidates[c_idx]
                r_c_distance = r_c_distance[c_idx]  # save only R-C-distance for selected C

            # update R-C-distances and mean for next heartbeat
            prev_r_c_distances.append(r_c_distance)
            if len(prev_r_c_distances) > self.window_c_correction:
                prev_r_c_distances.pop(0)
            mean_prev_r_c_distance = np.mean(prev_r_c_distances)

            # save C-point (and C-candidates) to result property
            c_points.loc[idx, "c_point_sample"] = (
                selected_c + heartbeat_start
            )  # get C-point relative to complete signal

        if len(heartbeats_no_c) > 0:
            c_points.loc[heartbeats_no_c, "nan_reason"] = "no_c_detected"
            missing_str = f"No valid C-point detected in {len(heartbeats_no_c)} heartbeats ({heartbeats_no_c})"
            if self.handle_missing_events == "warn":
                warnings.warn(missing_str)
            elif self.handle_missing_events == "raise":
                raise EventExtractionError(missing_str)

        _assert_is_dtype(c_points, pd.DataFrame)
        _assert_has_columns(c_points, [["c_point_sample", "nan_reason"]])
        c_points = c_points.astype({"c_point_sample": "Int64", "nan_reason": "object"})
        assert_sample_columns_int(c_points)

        self.points_ = c_points
        return self
