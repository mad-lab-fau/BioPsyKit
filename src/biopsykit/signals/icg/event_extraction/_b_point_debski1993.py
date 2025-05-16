import warnings

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
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

__all__ = ["BPointExtractionDebski1993SecondDerivative"]


class BPointExtractionDebski1993SecondDerivative(BaseBPointExtraction, CanHandleMissingEventsMixin):
    """B-point extraction algorithm by Debski et al. (1993) based on the reversal of dZ^2/dt^2 before the C-point.

    This algorithm extracts B-points based on the last reversal (local minimum) of the second derivative
    of the ICG signal before the C-point.

    For more information, see [Deb93]_.

    References
    ----------
    .. [Deb93] Debski, T. T., Zhang, Y., Jennings, J. R., & Kamarck, T. W. (1993). Stability of cardiac impedance
        measures: Aortic opening (B-point) detection and scoring. Biological Psychology, 36(1-2), 63-74.
        https://doi.org/10.1016/0301-0511(93)90081-I

    """

    # input parameters
    correct_outliers: Parameter[bool]

    def __init__(self, correct_outliers: bool = False, handle_missing_events: HANDLE_MISSING_EVENTS = "warn"):
        """Initialize new ``BPointExtractionDebski1993SecondDerivative`` instance.

        Parameters
        ----------
        correct_outliers : bool, optional
            Indicates whether to perform outlier correction (True) or not (False). Default: False
        handle_missing_events : one of {"warn", "raise", "ignore"}, optional
            How to handle failing event extraction. Can be one of:
                * "warn": issue a warning and set the event to NaN
                * "raise": raise an ``EventExtractionError``
                * "ignore": ignore the error and continue with the next event
            Default: "warn"


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
        sampling_rate_hz: float | None,  # noqa: ARG002
    ):
        """Extract B-points from given ICG derivative signal.

        This algorithm extracts B-points based on the last reversal (local minimum) of the second derivative of the
        ICG signal before the C-point.

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
        # sanitize input
        is_icg_raw_dataframe(icg)
        is_heartbeat_segmentation_dataframe(heartbeats)
        is_c_point_dataframe(c_points)
        icg = sanitize_input_dataframe_1d(icg, column="icg_der")
        icg = icg.squeeze()

        # Create the b_point Dataframe. Use the heartbeats id as index
        b_points = pd.DataFrame(index=heartbeats.index, columns=["b_point_sample", "nan_reason"])

        # get the r_peak locations from the heartbeats dataframe and search for entries containing NaN
        r_peaks = heartbeats["r_peak_sample"]
        check_r_peaks = pd.isna(r_peaks)

        # get the c_point locations from the c_points dataframe and search for entries containing NaN
        c_points = c_points["c_point_sample"]
        check_c_points = pd.isna(c_points)

        # Compute the second derivative of the ICG-signal
        icg_2nd_der = np.gradient(icg)

        # go through each R-C interval independently and search for the local minima
        for idx, data in heartbeats.iterrows():
            # check if r_peaks/c_points contain NaN. If this is the case, set the b_point to NaN and continue
            # with the next iteration
            missing_str = None
            if check_r_peaks[idx]:
                b_points.loc[idx, "b_point_sample"] = np.nan
                b_points.loc[idx, "nan_reason"] = "r_peak_nan"
                missing_str = f"The r_peak contains NaN at position {idx}! B-Point was set to NaN."
            if check_c_points[idx]:
                b_points.loc[idx, "b_point_sample"] = np.nan
                b_points.loc[idx, "nan_reason"] = "c_point_nan"
                missing_str = f"The c_point contains NaN at position {idx}! B-Point was set to NaN."

            if missing_str is not None:
                if self.handle_missing_events == "warn":
                    warnings.warn(missing_str)
                elif self.handle_missing_events == "raise":
                    raise EventExtractionError(missing_str)
                continue

            b_point = self._b_point_core_extraction(icg_2nd_der, r_peaks[idx], c_points[idx])

            if np.isnan(b_point):
                if self.correct_outliers:
                    b_point = data["r_peak_sample"]
                b_points.loc[idx, "nan_reason"] = "no_local_minimum"
            # Add the detected B-point to the b_points Dataframe
            b_points.loc[idx, "b_point_sample"] = b_point

        b_points = b_points.astype({"b_point_sample": "Int64", "nan_reason": "object"})
        is_b_point_dataframe(b_points)

        self.points_ = b_points

        return self

    @staticmethod
    def _b_point_core_extraction(
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
