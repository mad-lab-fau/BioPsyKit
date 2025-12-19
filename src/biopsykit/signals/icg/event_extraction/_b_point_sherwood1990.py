import warnings

import numpy as np
import pandas as pd

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


class BPointExtractionSherwood1990(BaseBPointExtraction, CanHandleMissingEventsMixin):
    """B-point extraction algorithm by Sherwood et al. (1990).

    This algorithm extracts B-points based on the last zero crossing of the ICG signal before the C-point.

    For more information, see [She90]_.

    References
    ----------
    .. [She90] Sherwood, A., Allen, M. T., Fahrenberg, J., Kelsey, R. M., Lovallo, W. R., & Doornen, L. J. P. (1990).
        Methodological Guidelines for Impedance Cardiography. Psychophysiology, 27(1), 1-23.
        https://doi.org/10.1111/j.1469-8986.1990.tb02171.x

    """

    def __init__(self, handle_missing_events: HANDLE_MISSING_EVENTS = "warn"):
        """Initialize new ``BPointExtractionSherwood1990`` instance.

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

    def extract(
        self,
        *,
        icg: IcgRawDataFrame,
        heartbeats: HeartbeatSegmentationDataFrame,
        c_points: CPointDataFrame,
        sampling_rate_hz: float | None,  # noqa: ARG002
    ):
        """Extract B-points from given ICG derivative signal.

        This algorithm extracts B-points based on the last zero crossing of the ICG signal before the C-point.

        The results are stored in the ``points_`` attribute of this class.

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
        # sanitize input
        is_icg_raw_dataframe(icg)
        is_heartbeat_segmentation_dataframe(heartbeats)
        is_c_point_dataframe(c_points)
        icg = sanitize_input_dataframe_1d(icg, column="icg_der")
        icg = icg.squeeze()

        # Create the b_point Dataframe. Use the heartbeats id as index
        b_points = pd.DataFrame(index=heartbeats.index, columns=["b_point_sample", "nan_reason"])

        # get the c_point locations from the c_points dataframe and search for entries containing NaN
        c_points = c_points["c_point_sample"]
        check_c_points = pd.isna(c_points)

        # get zero crossings of icg
        zero_crossings = np.where(np.diff(np.signbit(icg)))[0]

        # go through each R-C interval independently and search for the local minima
        for idx, data in heartbeats.iterrows():
            # check if r_peaks/c_points contain NaN. If this is the case, set the b_point to NaN and continue
            # with the next iteration
            if check_c_points[idx]:
                b_points.loc[idx, "b_point_sample"] = np.nan
                b_points.loc[idx, "nan_reason"] = "c_point_nan"
                missing_str = f"The c_point contains NaN at position {idx}! B-Point was set to NaN."
                if self.handle_missing_events == "warn":
                    warnings.warn(missing_str)
                elif self.handle_missing_events == "raise":
                    raise EventExtractionError(missing_str)
                continue

            # get the closest zero crossing *before* the C-point
            c_point = c_points[idx]
            zero_crossings_diff = zero_crossings - c_point
            zero_crossings_diff = zero_crossings_diff[zero_crossings_diff < 0]
            zero_crossing_idx = np.argmax(zero_crossings_diff)

            b_point = zero_crossings[zero_crossing_idx]
            # assert that b_point is within the R-C interval
            if not (data["r_peak_sample"] < b_point < c_point):
                b_point = np.nan
                b_points.loc[idx, "nan_reason"] = "no_zero_crossing"

            # Add the detected B-point to the b_points Dataframe
            b_points.loc[idx, "b_point_sample"] = b_point

        b_points = b_points.astype({"b_point_sample": "Int64", "nan_reason": "object"})
        is_b_point_dataframe(b_points)

        self.points_ = b_points

        return self
