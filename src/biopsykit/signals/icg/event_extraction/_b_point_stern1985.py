import warnings

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS, CanHandleMissingEventsMixin
from biopsykit.signals.icg.event_extraction._base_b_point_extraction import BaseBPointExtraction
from biopsykit.utils.array_handling import sanitize_input_dataframe_1d
from biopsykit.utils.dtypes import (
    CPointDataFrame,
    HeartbeatSegmentationDataFrame,
    IcgRawDataFrame,
    is_b_point_dataframe,
    is_c_point_dataframe,
    is_heartbeat_segmentation_dataframe,
    is_icg_raw_dataframe,
)
from biopsykit.utils.exceptions import EventExtractionError

__all__ = ["BPointExtractionStern1985"]


class BPointExtractionStern1985(BaseBPointExtraction, CanHandleMissingEventsMixin):
    """B-point extraction algorithm by Stern et al. (1985).

    This algorithm extracts B-points based on the last local minimum of the dZ/dt curve before the C-point.

    For more information, see [Ste85]_.

    References
    ----------
    .. [Ste85] Stern, H. C., Wolf, G. K., & Belz, G. G. (1985). Comparative measurements of left ventricular ejection
        time by mechano-, echo- and electrical impedance cardiography. Arzneimittel-Forschung, 35(10), 1582-1586.

    """

    def __init__(self, handle_missing_events: HANDLE_MISSING_EVENTS = "warn"):
        """Initialize new ``BPointExtractionStern1985`` instance.

        Parameters
        ----------
        handle_missing_events : one of {"warn", "raise", "ignore"}, optional
            How to handle failing event extraction. Can be one of:
                * "warn": issue a warning and set the event to NaN
                * "raise": raise an ``EventExtractionError``
                * "ignore": ignore the error and continue with the next event
            Default: "warn"

        """
        super().__init__(handle_missing_events=handle_missing_events)

    # @make_action_safe
    def extract(
        self,
        *,
        icg: IcgRawDataFrame,
        heartbeats: HeartbeatSegmentationDataFrame,
        c_points: CPointDataFrame,
        sampling_rate_hz: float | None,  # noqa: ARG002
    ):
        """Extract B-points from given ICG derivative signal.

        This algorithm extracts B-points based on the last local minimum of the dZ/dt curve before the C-point.

        The results are saved in the ``points_`` attribute of the super class.

        Parameters
        ----------
        icg : :class:`~pandas.DataFrame`
            ICG derivative signal
        heartbeats : :class:`~pandas.DataFrame`
            Segmented heartbeats. Each row contains start, end, and R-peak location (in samples
            from beginning of signal) of that heartbeat, index functions as id of heartbeat
        c_points : :class:`~pandas.DataFrame`
            Extracted C-points. Each row contains the C-point location (in samples from beginning of signal) for each
            heartbeat, index functions as id of heartbeat. C-point locations can be NaN if no C-points were detected
            for certain heartbeats
        sampling_rate_hz : int
            sampling rate of ICG derivative signal in hz

        Returns
        -------
            self

        Raises
        ------
        :exc:`~biopsykit.utils.exceptions.EventExtractionError`
            If the event extraction fails and ``handle_missing`` is set to "raise"

        """
        self._check_valid_missing_handling()
        is_icg_raw_dataframe(icg)
        is_heartbeat_segmentation_dataframe(heartbeats)
        is_c_point_dataframe(c_points)
        # sanitize input
        icg = sanitize_input_dataframe_1d(icg, column="icg_der")
        icg = icg.squeeze()

        # Create the b_point Dataframe. Use the heartbeats id as index
        b_points = pd.DataFrame(index=heartbeats.index, columns=["b_point_sample", "nan_reason"])

        # get the c_point locations from the c_points dataframe and search for entries containing NaN
        c_points = c_points["c_point_sample"]
        check_c_points = pd.isna(c_points)

        # Compute the second derivative of the ICG-signal
        icg_2nd_der = np.gradient(icg)
        icg_der_zero_crossings = np.where(np.diff(np.signbit(icg_2nd_der)))[0]

        # go through each heartbeat independently and search for the local minima
        for idx, data in heartbeats.iterrows():
            heartbeat_start = data["start_sample"]

            # check c_point is NaN. If this is the case, set the b_point to NaN and continue with the next iteration
            if check_c_points[idx]:
                b_points.loc[idx, "b_point_sample"] = np.nan
                b_points.loc[idx, "nan_reason"] = "c_point_nan"
                missing_str = f"The c_point is NaN at position {idx}! B-Point was set to NaN."
                if self.handle_missing_events == "warn":
                    warnings.warn(missing_str)
                elif self.handle_missing_events == "raise":
                    raise EventExtractionError(missing_str)
                continue

            # check if there are zero crossings in the interval between start of the heartbeat and the C-point
            # we subtract 1 to avoid the C-point itself since the zero crossing should be *before* the C-point and the
            # zero crossings are computed in a way that the sample before the zero crossing is returned
            zero_crossings_heartbeat = icg_der_zero_crossings[
                (icg_der_zero_crossings >= heartbeat_start) & (icg_der_zero_crossings < (c_points[idx] - 1))
            ]

            # if there are no zero crossings in the interval, set B-point to NaN
            if len(zero_crossings_heartbeat) == 0:
                b_points.loc[idx, "b_point_sample"] = np.nan
                b_points.loc[idx, "nan_reason"] = "no_local_minimum"
                continue
            # get the closest zero crossing *before* the C-point
            zero_crossings_diff = zero_crossings_heartbeat - c_points[idx]
            zero_crossings_diff = zero_crossings_diff[zero_crossings_diff < 0]
            zero_crossing_idx = np.argmax(zero_crossings_diff)

            b_point = zero_crossings_heartbeat[zero_crossing_idx]

            b_points.loc[idx, "b_point_sample"] = b_point

        b_points = b_points.astype({"b_point_sample": "Int64", "nan_reason": "object"})
        is_b_point_dataframe(b_points)

        self.points_ = b_points

        return self

    def _b_point_core_extraction(
        self,
        icg_2nd_der: pd.Series,
        r_peak: pd.Series,
        c_point: pd.Series,
    ):
        # set the borders of the interval between the R-Peak and the C-Point
        start_r_c = r_peak
        end_r_c = c_point

        # Select the specific interval in the second derivative of the ICG-signal
        icg_search_window = icg_2nd_der[start_r_c : (end_r_c + 1)]

        # Compute the local minima in this interval
        # icg_min = argrelmin(icg_search_window)
        icg_min = find_peaks(-icg_search_window)[0]
        # print(icg_min)

        # Compute the distance between the C-point and the minima of the interval and select the entry with
        # the minimal distance as B-point
        if len(icg_min) >= 1:
            distance = end_r_c - icg_min
            b_point_idx = distance.argmin()
            b_point = icg_min[b_point_idx]
            # Compute the absolute sample position of the local B-point
            b_point = b_point + start_r_c
        else:
            # If there is no minima set the B-Point to NaN
            b_point = np.nan

        return b_point
