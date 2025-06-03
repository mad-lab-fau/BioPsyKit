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

__all__ = ["BPointExtractionDrost2022"]


class BPointExtractionDrost2022(BaseBPointExtraction, CanHandleMissingEventsMixin):
    """B-point extraction algorithm by Drost et al. (2022).

    This algorithm extracts B-points based on the maximum distance of the dZ/dt curve and a straight line fitted
    between the C-Point and the Point on the dZ/dt curve 150 ms before the C-Point.

    For more information, see [Dro22]_.

    References
    ----------
    .. [Dro22] Drost, L., Finke, J. B., Port, J., & Schächinger, H. (2022). Comparison of TWA and PEP as indices of
        a2- and ß-adrenergic activation. Psychopharmacology. https://doi.org/10.1007/s00213-022-06114-8

    """

    def __init__(self, handle_missing_events: HANDLE_MISSING_EVENTS = "warn"):
        """Initialize new ``BPointExtractionDrost2022`` instance.

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
        sampling_rate_hz: float,
    ):
        """Extract B-points from given ICG derivative signal.

        This algorithm extracts B-points based on the maximum distance of the dZ/dt curve and a straight line
        fitted between the C-Point and the Point on the dZ/dt curve 150 ms before the C-Point.

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
        icg = sanitize_input_dataframe_1d(icg, column="icg_der")
        icg = icg.squeeze()

        # Create the b_point Dataframe. Use the heartbeats id as index
        b_points = pd.DataFrame(index=heartbeats.index, columns=["b_point_sample", "nan_reason"])

        # get the c_point locations from the c_points dataframe
        c_points = c_points["c_point_sample"]

        # iterate over each heartbeat
        for idx, _data in heartbeats.iterrows():
            # Get the C-Point location at the current heartbeat id
            c_point = c_points[idx]

            if pd.isna(c_point):
                b_points.loc[idx, "b_point_sample"] = np.nan
                b_points.loc[idx, "nan_reason"] = "c_point_nan"
                continue

            # Calculate the start position of the straight line (150 ms before the C-Point) and ensure that the
            # start position is not negative
            line_start = max(c_point - int((150 / 1000) * sampling_rate_hz), 0)

            # Calculate the values of the straight line
            line_values = self._get_straight_line(line_start, icg.iloc[line_start], c_point, icg.iloc[c_point])

            # Get the interval of the cleaned ICG-signal in the range of the straight line
            signal_clean_interval = icg.iloc[line_start:c_point].squeeze()

            # Calculate the distance between the straight line and the cleaned ICG-signal
            distance = line_values["result"].to_numpy() - signal_clean_interval.to_numpy()

            # Calculate the location of the maximum distance and transform the index relative to the complete signal
            # to obtain the B-Point location
            b_point = line_start + np.argmax(distance)

            b_points.loc[idx, "b_point_sample"] = b_point

        num_nan = b_points["b_point_sample"].isna().sum()
        if num_nan > 0:
            idx_nan = b_points["b_point_sample"].isna()
            idx_nan = list(b_points.index[idx_nan])

            missing_str = (
                f"The C-point contains NaN at heartbeats {idx_nan}! The index of the B-points were also set to NaN."
            )
            if self.handle_missing_events == "warn":
                warnings.warn(missing_str)
            elif self.handle_missing_events == "raise":
                raise EventExtractionError(missing_str)

        b_points = b_points.astype({"b_point_sample": "Int64", "nan_reason": "object"})
        is_b_point_dataframe(b_points)

        self.points_ = b_points
        return self

    @staticmethod
    def _get_straight_line(start_x: int, start_y: float, c_x: int, c_y: float):
        """Compute the values of a straight line fitted between the C-Point and the point 150 ms before the C-Point.

        Parameters
        ----------
        start_x: int
            index of the point 150 ms before the C-Point
        start_y: float
            value of the point 150 ms before the C-Point
        c_x: int
            index of the C-Point
        c_y: float
            value of the C-Point

        Returns
        -------
        :class:`~pandas.DataFrame`
            DataFrame containing the values of the straight line for each index between the C-Point and the point
            150 ms before the C-Point

        """
        # Compute the slope of the straight line
        start_y = float(start_y)
        c_y = float(c_y)
        slope = float((c_y - start_y) / (c_x - start_x))

        # Get the sample positions where we want to calculate the values of the straight line
        index = np.arange(0, (c_x - start_x), 1)
        line_values = pd.DataFrame(index=index, columns=["result"])

        # Compute the values of the straight line for each index
        line_values["result"] = (index * slope) + start_y

        return line_values
