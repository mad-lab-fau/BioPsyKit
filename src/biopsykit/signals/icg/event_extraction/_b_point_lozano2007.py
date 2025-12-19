import numpy as np
import pandas as pd
from tpcp import Parameter

from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS, CanHandleMissingEventsMixin
from biopsykit.signals.icg.event_extraction._base_b_point_extraction import BaseBPointExtraction

__all__ = ["BPointExtractionLozano2007LinearRegression", "BPointExtractionLozano2007QuadraticRegression"]

from biopsykit.utils.dtypes import (
    CPointDataFrame,
    HeartbeatSegmentationDataFrame,
    IcgRawDataFrame,
    is_b_point_dataframe,
    is_c_point_dataframe,
    is_heartbeat_segmentation_dataframe,
    is_icg_raw_dataframe,
)


class BPointExtractionLozano2007LinearRegression(BaseBPointExtraction, CanHandleMissingEventsMixin):
    """B-point extraction algorithm by Lozano et al. (2007) based on linear regression of R-C interval.

    This algorithm extracts B-points based on the linear regression of the relationship between the R-C interval
    and the B-point.

    For more information, see [Loz07]_.

    References
    ----------
    .. [Loz07] Lozano, D. L., Norman, G., Knox, D., Wood, B. L., Miller, B. D., Emery, C. F., & Berntson, G. G. (2007).
        Where to B in dZ/dt. Psychophysiology, 44(1), 113-119. https://doi.org/10.1111/j.1469-8986.2006.00468.x

    """

    # input parameters
    moving_average_window: Parameter[int]

    def __init__(self, moving_average_window: int = 1, handle_missing_events: HANDLE_MISSING_EVENTS = "warn"):
        """Initialize new ``BPointExtractionLozano2007LinearRegression`` instance.

        Parameters
        ----------
        moving_average_window : int, optional
            Window size for moving average filter (in heartbeats, centered around the current heartbeat)
            to compute the R-C interval. Default: 1 (no moving average)
        handle_missing_events : one of {"warn", "raise", "ignore"}, optional
            How to handle failing event extraction. Can be one of:
                * "warn": issue a warning and set the event to NaN
                * "raise": raise an ``EventExtractionError``
                * "ignore": ignore the error and continue with the next event
            Default: "warn"

        """
        super().__init__(handle_missing_events=handle_missing_events)
        self.moving_average_window = moving_average_window

    def extract(
        self,
        *,
        icg: IcgRawDataFrame,
        heartbeats: HeartbeatSegmentationDataFrame,
        c_points: CPointDataFrame,
        sampling_rate_hz: float,
    ):
        """Extract B-points from given ICG derivative signal.

        This algorithm extracts B-points using linear regression based on the relationship between
        the R-C interval and the B-point.

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

        # result dfs
        b_points = pd.DataFrame(index=heartbeats.index, columns=["b_point_sample", "nan_reason"])

        # used subsequently to store ids of heartbeats where no B was detected because there was no C
        # (Bs should always be found, since they are set to the max of the 3rd derivative, and there is always a max)
        heartbeats_no_c_b = []

        # search B-point for each heartbeat of the given signal
        for idx, _data in heartbeats.iterrows():
            if self.moving_average_window == 1:
                c_point_sample = c_points.loc[[idx], "c_point_sample"]
                r_peak_sample = heartbeats.loc[[idx], "r_peak_sample"]
            else:
                window_width = self.moving_average_window // 2
                start_idx = heartbeats.index[max(0, idx - window_width)]
                end_idx = heartbeats.index[min(len(heartbeats) - 1, idx + window_width + 1)]
                c_point_sample = c_points.loc[start_idx:end_idx, "c_point_sample"].dropna()
                r_peak_sample = heartbeats.loc[start_idx:end_idx, "r_peak_sample"].dropna()

            # C-point can be NaN, then, extraction of B is not possible, so B is set to NaN
            if pd.isna(c_point_sample).any():
                heartbeats_no_c_b.append(idx)
                b_points.loc[idx, "b_point_sample"] = np.nan
                b_points.loc[idx, "nan_reason"] = "c_point_nan"
                continue

            current_r_peak = heartbeats.loc[idx, "r_peak_sample"]
            # get the R-C interval in ms
            r_c_interval_ms = np.mean((c_point_sample - r_peak_sample) / sampling_rate_hz * 1000)
            if pd.isna(r_c_interval_ms):
                b_points.loc[idx, "b_point_sample"] = np.nan
                b_points.loc[idx, "nan_reason"] = "no_r_c_interval"
                continue

            b_point_interval_ms = 0.55 * r_c_interval_ms + 4.45
            b_point_interval_sample = int((b_point_interval_ms * sampling_rate_hz) / 1000)
            b_point_sample = current_r_peak + b_point_interval_sample

            b_points.loc[idx, "b_point_sample"] = b_point_sample

        b_points = b_points.astype({"b_point_sample": "Int64", "nan_reason": "object"})
        is_b_point_dataframe(b_points)

        self.points_ = b_points
        return self


class BPointExtractionLozano2007QuadraticRegression(BaseBPointExtraction, CanHandleMissingEventsMixin):
    """B-point extraction algorithm by Lozano et al. (2007) [1]_ based on quadratic regression of R-C interval.

    This algorithm extracts B-points based on the quadratic regression of the relationship between the R-C interval
    and the B-point.

    References
    ----------
    .. [1] Lozano, D. L., Norman, G., Knox, D., Wood, B. L., Miller, B. D., Emery, C. F., & Berntson, G. G. (2007).
        Where to B in dZ/dt. Psychophysiology, 44(1), 113-119. https://doi.org/10.1111/j.1469-8986.2006.00468.x

    """

    def __init__(self, moving_average_window: int = 1, handle_missing_events: HANDLE_MISSING_EVENTS = "warn"):
        """Initialize new ``BPointExtractionLozano2007QuadraticRegression`` instance.

        Parameters
        ----------
        moving_average_window : int, optional
            Window size for moving average filter (in heartbeats, centered around the current heartbeat)
            to compute the R-C interval. Default: 1 (no moving average)
        handle_missing_events : one of {"warn", "raise", "ignore"}, optional
            How to handle failing event extraction. Can be one of:
                * "warn": issue a warning and set the event to NaN
                * "raise": raise an ``EventExtractionError``
                * "ignore": ignore the error and continue with the next event
            Default: "warn"

        """
        super().__init__(handle_missing_events=handle_missing_events)
        self.moving_average_window = moving_average_window

    def extract(
        self,
        *,
        icg: IcgRawDataFrame,
        heartbeats: HeartbeatSegmentationDataFrame,
        c_points: CPointDataFrame,
        sampling_rate_hz: float,
    ):
        """Extract B-points from given ICG derivative signal.

        This algorithm extracts B-points using quadratic regression based on the relationship between
        the R-C interval and the B-point.

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

        # result dfs
        b_points = pd.DataFrame(index=heartbeats.index, columns=["b_point_sample", "nan_reason"])

        # used subsequently to store ids of heartbeats where no B was detected because there was no C
        # (Bs should always be found, since they are set to the max of the 3rd derivative, and there is always a max)
        heartbeats_no_c_b = []

        # search B-point for each heartbeat of the given signal
        for idx, _data in heartbeats.iterrows():
            if self.moving_average_window == 1:
                c_point_sample = c_points.loc[[idx], "c_point_sample"]
                r_peak_sample = heartbeats.loc[[idx], "r_peak_sample"]
            else:
                window_width = self.moving_average_window // 2
                start_idx = heartbeats.index[max(0, idx - window_width)]
                end_idx = heartbeats.index[min(len(heartbeats) - 1, idx + window_width + 1)]
                c_point_sample = c_points.loc[start_idx:end_idx, "c_point_sample"].dropna()
                r_peak_sample = heartbeats.loc[start_idx:end_idx, "r_peak_sample"].dropna()

            # C-point can be NaN, then, extraction of B is not possible, so B is set to NaN
            if pd.isna(c_point_sample).any():
                heartbeats_no_c_b.append(idx)
                b_points.loc[idx, "b_point_sample"] = np.nan
                b_points.loc[idx, "nan_reason"] = "c_point_nan"
                continue

            current_r_peak = heartbeats.loc[idx, "r_peak_sample"]
            # get the R-C interval in ms
            r_c_interval_ms = np.mean((c_point_sample - r_peak_sample) / sampling_rate_hz * 1000)
            if pd.isna(r_c_interval_ms):
                b_points.loc[idx, "b_point_sample"] = np.nan
                b_points.loc[idx, "nan_reason"] = "no_r_c_interval"
                continue
            b_point_interval_ms = -0.0032 * r_c_interval_ms**2 + 1.233 * r_c_interval_ms - 31.59
            b_point_interval_sample = int((b_point_interval_ms * sampling_rate_hz) / 1000)
            b_point_sample = current_r_peak + b_point_interval_sample

            b_points.loc[idx, "b_point_sample"] = b_point_sample

        b_points = b_points.astype({"b_point_sample": "Int64", "nan_reason": "object"})
        is_b_point_dataframe(b_points)

        self.points_ = b_points
        return self
