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

__all__ = ["BPointExtractionPale2021"]


class BPointExtractionPale2021(BaseBPointExtraction, CanHandleMissingEventsMixin):
    """B-point extraction algorithm by Pale et al. (2021).

    This algorithm extracts B-points by first determining a search window based on the C-point location and the
    C-point amplitude in the dZ/dt signal. Afterward, the algorithms searches for either the local minimum closest to
    the C-point or the first point at which the slope of the signal exceeds a certain threshold. If no criterion is met,
    the search is repeated with a less strict slope threshold. If still no B-point is found, the algorithm
    returns the signal minimum in the cardiac cycle before the C-point.

    For more information, see [Pal21]_.

    References
    ----------
    .. [Pal21] Pale, U., Muller, N., Arza, A., & Atienza, D. (2021). ReBeatICG: Real-time Low-Complexity Beat-to-beat
        Impedance Cardiogram Delineation Algorithm. 2021 43rd Annual International Conference of the IEEE Engineering
        in Medicine & Biology Society (EMBC), 5618-5624. https://doi.org/10.1109/EMBC46164.2021.9630170

    """

    c_point_amplitude_fraction: float
    b_point_slope_threshold_01: float
    b_point_slope_threshold_02: float

    def __init__(
        self,
        c_point_amplitude_fraction: float = 0.5,
        b_point_slope_threshold_01: float = 0.11,
        b_point_slope_threshold_02: float = 0.08,
        handle_missing_events: HANDLE_MISSING_EVENTS = "warn",
    ):
        """Initialize new ``BPointExtractionPale2021`` instance.

        Parameters
        ----------
        handle_missing_events : one of {"warn", "raise", "ignore"}, optional
            How to handle failing event extraction. Can be one of:
                * "warn": issue a warning and set the event to NaN
                * "raise": raise an ``EventExtractionError``
                * "ignore": ignore the error and continue with the next event
            Default: "warn"

        """
        self.c_point_amplitude_fraction = c_point_amplitude_fraction
        self.b_point_slope_threshold_01 = b_point_slope_threshold_01
        self.b_point_slope_threshold_02 = b_point_slope_threshold_02
        super().__init__(handle_missing_events=handle_missing_events)

    def extract(
        self,
        *,
        icg: IcgRawDataFrame,
        heartbeats: HeartbeatSegmentationDataFrame,
        c_points: CPointDataFrame,
        sampling_rate_hz: float,
    ):
        """Extract B-points from given ICG derivative signal.

        This algorithm extracts B-points by transforming the ICG signal using a weighted time window applied to the
        segment preceding the maximal ICG peak (C-point). This transformation amplifies the characteristics of the
        B-point, facilitating B-point identification.

        Parameters
        ----------
        icg : IcgRawDataFrame
            The raw ICG signal data.
        heartbeats : HeartbeatSegmentationDataFrame
            The heartbeat segmentation data.
        c_points : CPointDataFrame
            The C-point data.
        sampling_rate_hz : float
            The sampling rate of the ICG signal in Hz.

        Returns
        -------
        BPointDataFrame
            The extracted B-point data.

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

        icg_2nd_der = np.gradient(icg)

        # iterate over each heartbeat
        for idx, data in heartbeats.iterrows():
            # Get the C-Point location at the current heartbeat id
            c_point = c_points[idx]

            if pd.isna(c_point):
                b_points.loc[idx, "b_point_sample"] = np.nan
                b_points.loc[idx, "nan_reason"] = "c_point_nan"
                continue

            icg_slice = icg.iloc[data["start_sample"] : c_point]

            # Calculate the search window based on the C-point location and amplitude
            # start of the search window is 80 ms before the C-point
            search_window_start = c_point - int(0.08 * sampling_rate_hz)
            # end of the search window is the closest point before the C-point with amplitude less than
            # c_point_amplitude_fraction * c_point
            c_point_amplitude = icg.iloc[c_point] * self.c_point_amplitude_fraction

            search_window_end = np.where(icg.iloc[search_window_start:c_point] < c_point_amplitude)[0]
            # If no point is found, use the C-point as the end of the search window; otherwise, use the last point
            # before the C-point that meets the condition
            search_window_end = c_point if search_window_end.size == 0 else search_window_end[-1] + search_window_start

            # B_min is the minimum of the signal in the search window
            b_point_min = data["start_sample"] + np.argmin(icg_slice)

            if (search_window_end - search_window_start) <= 2:
                # If the search window is too small, set the B-point to the minimum of the signal in the search window
                b_points.loc[idx, "b_point_sample"] = b_point_min
                continue

            # slice derivative to the search window
            icg_2nd_der_slice = icg_2nd_der[search_window_start:search_window_end]

            # candidate 1: search for local minima in the second derivative
            zero_crossings = np.where(np.diff(np.signbit(icg_2nd_der_slice)))[0]
            # check if it's a local minimum => the value of the derivative at the zero crossing must be positive
            zero_crossings = zero_crossings[np.gradient(icg_2nd_der_slice)[zero_crossings] > 0]

            # candidate 2: search for the first point at which the slope exceeds the threshold; the slope is already
            # calculated in the derivative
            slope_exceeds_threshold = np.where(icg_2nd_der_slice > self.b_point_slope_threshold_01)[0]
            # if no slope exceeds the threshold, use the second threshold
            if slope_exceeds_threshold.size == 0:
                slope_exceeds_threshold = np.where(icg_2nd_der_slice > self.b_point_slope_threshold_02)[0]
            # concatenate and sort the candidates
            candidates = np.sort(np.concatenate((zero_crossings, slope_exceeds_threshold)))

            # if no candidates are found, use the minimum of the signal in the search window; otherwise, use the
            # candidate closest to the C-point
            b_point = b_point_min if candidates.size == 0 else search_window_start + candidates[-1]

            b_points.loc[idx, "b_point_sample"] = b_point

        idx_nan = b_points["b_point_sample"].isna()
        if idx_nan.sum() > 0:
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
