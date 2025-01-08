import warnings
from typing import Optional

import numpy as np
import pandas as pd
from tpcp import Parameter

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
    """Algorithm by Drost et al. (2022) to extract B-points based on the distance of the dZ/dt curve to a straight line.

    This algorithm extracts B-points based on the maximum distance of the dZ/dt curve and a straight line fitted
    between the C-Point and the Point on the dZ/dt curve 150 ms before the C-Point.

    References
    ----------
    Drost, L., Finke, J. B., Port, J., & Schächinger, H. (2022). Comparison of TWA and PEP as indices of a2- and
    ß-adrenergic activation. Psychopharmacology. https://doi.org/10.1007/s00213-022-06114-8

    """

    # input parameters
    correct_outliers: Parameter[bool]

    def __init__(self, correct_outliers: Optional[bool] = False, handle_missing_events: HANDLE_MISSING_EVENTS = "warn"):
        """Initialize new BPointExtractionDrost algorithm instance.

        Parameters
        ----------
        correct_outliers : bool
            Indicates whether to perform outlier correction (True) or not (False)
        """
        super().__init__(handle_missing_events=handle_missing_events)
        self.correct_outliers = correct_outliers

    # @make_action_safe
    def extract(
        self,
        *,
        icg: IcgRawDataFrame,
        heartbeats: HeartbeatSegmentationDataFrame,
        c_points: CPointDataFrame,
        sampling_rate_hz: float,
    ):
        """Extract B-points from given ICG cleaned signal.

        This algorithm extracts B-points based on the maximum distance of the dZ/dt curve and a straight line
        fitted between the C-Point and the Point on the dZ/dt curve 150 ms before the C-Point.

        The results are saved in the points_ attribute of the super class.

        Parameters
        ----------
        icg : :class:`~pandas.DataFrame`
            DataFrame containing the cleaned ICG signal
        heartbeats : :class:`~pandas.DataFrame`
            DataFrame containing one row per segmented heartbeat, each row contains start, end, and R-peak
            location (in samples from beginning of signal) of that heartbeat, index functions as id of heartbeat
        c_points : :class:`~pandas.DataFrame`
            DataFrame containing one row per segmented C-point, each row contains location
            (in samples from beginning of signal) of that C-point or NaN if the location of that C-point
            is not correct
        sampling_rate_hz : int
            Sampling rate of ECG signal in hz

        Returns
        -------
            self

        Raises
        ------
        :exc:`~biopsykit.utils.exceptions.EventExtractionError`
            If the C-Point contains NaN values and handle_missing is set to "raise"

        """
        self._check_valid_missing_handling()
        is_icg_raw_dataframe(icg)
        is_heartbeat_segmentation_dataframe(heartbeats)
        is_c_point_dataframe(c_points)
        icg = sanitize_input_dataframe_1d(icg, column="icg_der")
        icg = icg.squeeze()

        # Create the b_point Dataframe. Use the heartbeats id as index
        b_points = pd.DataFrame(index=heartbeats.index, columns=["b_point_sample", "nan_reason"])

        # get the c_point locations from the c_points dataframe and search for entries containing NaN
        check_c_points = pd.isna(c_points["c_point_sample"]).to_numpy()

        # iterate over each heartbeat
        for idx, _data in heartbeats.iterrows():
            # check if c_points contain NaN. If this is the case, set the b_point to NaN and continue
            # with the next iteration
            if check_c_points[idx]:
                b_points.loc[idx, "b_point_sample"] = np.NaN
                b_points.loc[idx, "nan_reason"] = "c_point_nan"
                continue
            # Get the C-Point location at the current heartbeat id
            c_point = c_points.loc[idx, "c_point_sample"]

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
