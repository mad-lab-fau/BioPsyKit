import warnings
from typing import Optional

import numpy as np
import pandas as pd
from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS
from biopsykit.signals.icg.event_extraction._base_c_point_extraction import BaseCPointExtraction
from biopsykit.utils._datatype_validation_helper import _assert_has_columns, _assert_is_dtype
from biopsykit.utils.exceptions import EventExtractionError
from scipy import signal
from tpcp import Parameter

__all__ = ["CPointExtractionScipyFindPeaks"]


class CPointExtractionScipyFindPeaks(BaseCPointExtraction):
    """Extract C-points from ICG derivative signal using scipy's find_peaks function."""

    # input parameters
    window_c_correction: Parameter[int]
    save_candidates: Parameter[bool]

    def __init__(self, window_c_correction: Optional[int] = 3, save_candidates: Optional[bool] = False):
        """Initialize the C-point extraction algorithm.

        Parameters
        ----------
        window_c_correction : int, optional
            how many preceding heartbeats are taken into account for C-point correction (using mean R-C-distance)
        save_candidates : bool, optional
            indicates whether only the selected C-point position (one per heartbeat) is saved in _points (False),
            or also all other C-candidates (True).

        """
        self.window_c_correction = window_c_correction
        self.save_candidates = save_candidates

    # @make_action_safe
    def extract(  # noqa: C901, PLR0912
        self,
        *,
        icg: pd.Series,
        heartbeats: pd.DataFrame,
        sampling_rate_hz: int,  # noqa: ARG002
        handle_missing: Optional[HANDLE_MISSING_EVENTS] = "warn",
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
        handle_missing : one of {"warn", "raise", "ignore"}, optional
            How to handle missing C-points (default: "warn").
            * "warn" : issue a warning and set C-point to NaN
            * "raise" : raise an :class:`~biopsykit.utils.exceptions.EventExtractionError`
            * "ignore" : ignore missing C-points

        Returns
        -------
        self

        """
        # result df
        c_points = pd.DataFrame(index=heartbeats.index, columns=["c_point_sample", "nan_reason"])
        if self.save_candidates:
            c_points = c_points.assign(c_point_candidates=np.empty((len(heartbeats.index), 0)).tolist())

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

            elif len(heartbeat_c_candidates) > 1:
                # take averaged R-C-distance over the 'window_c_correction' (default: 3) preceding heartbeats
                # calculate the absolute difference of R-C-distances for all C-candidates to this mean
                # (to check which of the C-candidates are most probably the wrongly detected Cs)
                distance_diff = np.abs(r_c_distance - mean_prev_r_c_distance)

                # choose the C-candidate with the smallest absolute difference in R-C-distance
                # (the one, where R-C-distance changed the least compared to previous heartbeats)
                c_idx = np.argmin(distance_diff)
                selected_c = heartbeat_c_candidates[c_idx]
                r_c_distance = r_c_distance[c_idx]  # save only R-C-distance for selected C

            else:
                warnings.warn("That should never happen!")
                selected_c = np.NaN

            # update R-C-distances and mean for next heartbeat
            prev_r_c_distances.append(r_c_distance)
            if len(prev_r_c_distances) > self.window_c_correction:
                prev_r_c_distances.pop(0)
            mean_prev_r_c_distance = np.mean(prev_r_c_distances)

            # save C-point (and C-candidates) to result property
            c_points.loc[idx, "c_point_sample"] = (
                selected_c + heartbeat_start
            )  # get C-point relative to complete signal
            if self.save_candidates:
                for c in heartbeat_c_candidates:
                    c_points.loc[idx, "c_point_candidates"].append(c + heartbeat_start)

        if len(heartbeats_no_c) > 0:
            c_points.loc[heartbeats_no_c, "nan_reason"] = "no_c_detected"
            missing_str = f"No valid C-point detected in {len(heartbeats_no_c)} heartbeats ({heartbeats_no_c})"
            if handle_missing == "warn":
                warnings.warn(missing_str)
            elif handle_missing == "raise":
                raise EventExtractionError(missing_str)

        _assert_is_dtype(c_points, pd.DataFrame)
        _assert_has_columns(c_points, [["c_point_sample", "nan_reason"]])

        self.points_ = c_points.convert_dtypes(infer_objects=True)
        return self
