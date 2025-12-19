import warnings

import numpy as np
import pandas as pd
from tpcp import Parameter

from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS, CanHandleMissingEventsMixin
from biopsykit.signals.icg.event_extraction._base_b_point_extraction import BaseBPointExtraction, bpoint_algo_docfiller
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

__all__ = [
    "BPointExtractionArbol2017IsoelectricCrossings",
    "BPointExtractionArbol2017SecondDerivative",
    "BPointExtractionArbol2017ThirdDerivative",
]


@bpoint_algo_docfiller
class BPointExtractionArbol2017IsoelectricCrossings(BaseBPointExtraction, CanHandleMissingEventsMixin):
    """B-point extraction algorithm by Arbol et al. (2017) based on isoelectric crossings.

    This algorithm extracts B-points based on the last crossing of the dZ/dt signal through the isoelectric line (i.e.,
    the mean of the dZ/dt signal in the cardiac cycle) before the C-point.

    For more information, see [Arb17]_.

    Parameters
    ----------
    %(base_parameters)s

    %(base_attributes)s

    References
    ----------
    .. [Arb17] Árbol, J. R., Perakakis, P., Garrido, A., Mata, J. L., Fernández-Santaella, M. C., & Vila, J. (2017).
        Mathematical detection of aortic valve opening (B point) in impedance cardiography: A comparison of three
        popular algorithms. Psychophysiology, 54(3), 350-357. https://doi.org/10.1111/psyp.12799

    """

    def __init__(self, handle_missing_events: HANDLE_MISSING_EVENTS = "warn"):
        """Initialize new ``BPointExtractionArbol2017IsoelectricCrossings`` algorithm instance.

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

        This algorithm extracts B-points based on the last isoelectric crossing before the C-point.

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

        # result dfs
        b_points = pd.DataFrame(index=heartbeats.index, columns=["b_point_sample", "nan_reason"])

        # used subsequently to store ids of heartbeats where no B was detected because there was no C
        # (Bs should always be found, since they are set to the max of the 3rd derivative, and there is always a max)
        heartbeats_no_c_b = []

        # search B-point for each heartbeat of the given signal
        for idx, data in heartbeats.iterrows():
            c_point_sample = c_points.loc[idx, "c_point_sample"]

            # C-point can be NaN, then, extraction of B is not possible, so B is set to NaN
            if pd.isna(c_point_sample):
                heartbeats_no_c_b.append(idx)
                b_points.loc[idx, "b_point_sample"] = np.nan
                b_points.loc[idx, "nan_reason"] = "c_point_nan"
                continue

            # slice the signal for the current heartbeat
            heartbeat_start = data["start_sample"]
            heartbeat_end = data["end_sample"]
            icg_heartbeat = icg.iloc[heartbeat_start:heartbeat_end]
            c_point = c_point_sample - heartbeat_start

            # compute the isoelectric line and subtract it from the signal
            isoelectric_line = np.mean(icg_heartbeat)
            icg_isoelectric = icg_heartbeat - isoelectric_line

            # compute the isoelectric crossings
            icg_isoelectric_crossings = np.where(np.diff(np.signbit(icg_isoelectric)))[0]

            # find the last isoelectric crossing *before* the C-point
            icg_isoelectric_crossings_diff = icg_isoelectric_crossings - c_point
            icg_isoelectric_crossings_diff = icg_isoelectric_crossings_diff[icg_isoelectric_crossings_diff < 0]
            if len(icg_isoelectric_crossings_diff) == 0:
                b_points.loc[idx, "b_point_sample"] = np.nan
                b_points.loc[idx, "nan_reason"] = "no_iso_crossing_before_c_point"
                continue
            icg_isoelectric_crossing_idx = np.argmax(icg_isoelectric_crossings_diff)

            b_point_idx = icg_isoelectric_crossings[icg_isoelectric_crossing_idx]
            b_point = b_point_idx + heartbeat_start

            b_points.loc[idx, "b_point_sample"] = b_point

        b_points = b_points.astype({"b_point_sample": "Int64", "nan_reason": "object"})
        is_b_point_dataframe(b_points)

        self.points_ = b_points
        return self


class BPointExtractionArbol2017SecondDerivative(BaseBPointExtraction, CanHandleMissingEventsMixin):
    """B-point extraction algorithm by Arbol et al. (2017) based on the second derivative of the ICG signal.

    This algorithm extracts B-points based on the maximum of the second derivative of the ICG signal in a 50ms window,
    starting 150ms before the C-point.

    For more information, see [Arb17]_.

    References
    ----------
    .. [Arb17] Árbol, J. R., Perakakis, P., Garrido, A., Mata, J. L., Fernández-Santaella, M. C., & Vila, J. (2017).
        Mathematical detection of aortic valve opening (B point) in impedance cardiography: A comparison of three
        popular algorithms. Psychophysiology, 54(3), 350-357. https://doi.org/10.1111/psyp.12799

    """

    # input parameters
    search_window_start_ms: Parameter[int]  # integer defining window start in ms
    window_size_ms: Parameter[int]  # integer defining window length in ms
    correct_outliers: Parameter[bool]

    def __init__(
        self,
        search_window_start_ms: int | None = 150,
        window_size_ms: int | None = 50,
        handle_missing_events: HANDLE_MISSING_EVENTS = "warn",
    ):
        """Initialize new ``BPointExtractionArbol2017SecondDerivative`` algorithm instance.

        Parameters
        ----------
        search_window_start_ms : int, optional
            Start of the search window in which the algorithm searches for the B-point, relative to the C-point.
            Default: 150 ms (see Arbol 2017)
        window_size_ms : str, int
            Size of the search window in which the algorithm searches for the B-point. Default: 50 ms (see Arbol 2017)
        handle_missing_events : one of {"warn", "raise", "ignore"}, optional
            How to handle failing event extraction. Can be one of:
                * "warn": issue a warning and set the event to NaN
                * "raise": raise an ``EventExtractionError``
                * "ignore": ignore the error and continue with the next event
            Default: "warn"

        """
        super().__init__(handle_missing_events=handle_missing_events)
        self.search_window_start_ms = search_window_start_ms
        self.window_size_ms = window_size_ms

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

        This algorithm extracts B-points based on the maximum of the second derivative of the ICG signal in a 50ms
        window, starting 150ms before the C-point.

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

        # result dfs
        b_points = pd.DataFrame(index=heartbeats.index, columns=["b_point_sample", "nan_reason"])

        # used subsequently to store ids of heartbeats where no B was detected because there was no C
        # (Bs should always be found, since they are set to the max of the 3rd derivative, and there is always a max)
        heartbeats_no_c_b = []
        # (but in case of wrongly detected Cs, the search window might be invalid, then no B can be found)
        heartbeats_no_b = []

        icg_2nd_der = np.gradient(icg)

        # search B-point for each heartbeat of the given signal
        for idx, _data in heartbeats.iterrows():
            c_point_sample = c_points.loc[idx, "c_point_sample"]

            # C-point can be NaN, then, extraction of B is not possible, so B is set to NaN
            if pd.isna(c_point_sample):
                heartbeats_no_c_b.append(idx)
                b_points.loc[idx, "b_point_sample"] = np.nan
                b_points.loc[idx, "nan_reason"] = "c_point_nan"
                continue

            # set window start according to specified method
            window_start = c_point_sample - int((self.search_window_start_ms / 1000) * sampling_rate_hz)
            window_end = window_start + int((self.window_size_ms / 1000) * sampling_rate_hz)

            # might happen for wrongly detected Cs (search window becomes invalid)
            if window_start < 0:
                heartbeats_no_b.append(idx)
                b_points.loc[idx, "b_point_sample"] = np.nan
                b_points.loc[idx, "nan_reason"] = "invalid_b_point_search_window"
                continue

            # find max in B window and calculate B-point relative to signal start
            b_window = icg_2nd_der[window_start:window_end]
            b_window_max = np.argmax(b_window)
            b_point_sample = window_start + b_window_max

            # inform user about missing B-points
            if len(heartbeats_no_c_b) > 0 or len(heartbeats_no_b) > 0:
                nan_rows = b_points[b_points["b_point_sample"].isna()]
                n = len(nan_rows)
                nan_rows = nan_rows.drop(index=heartbeats_no_c_b)
                nan_rows = nan_rows.drop(index=heartbeats_no_b)

                missing_str = (
                    f"No B-point detected in {n} heartbeats:\n"
                    f"- For heartbeats {heartbeats_no_c_b} no B point could be extracted, "
                    f"because there was no C point\n"
                    f"- For heartbeats {heartbeats_no_b} the search window was invalid probably due to "
                    f"wrongly detected C points\n"
                    f"- for heartbeats {nan_rows.index.to_numpy()} apparently also no B point was found "
                    f"for some other reasons"
                )

                if self.handle_missing_events == "warn":
                    warnings.warn(missing_str)
                elif self.handle_missing_events == "raise":
                    raise EventExtractionError(missing_str)

            b_points.loc[idx, "b_point_sample"] = b_point_sample

        b_points = b_points.astype({"b_point_sample": "Int64", "nan_reason": "object"})
        is_b_point_dataframe(b_points)

        self.points_ = b_points
        return self


class BPointExtractionArbol2017ThirdDerivative(BaseBPointExtraction, CanHandleMissingEventsMixin):
    """B-point extraction algorithm by Arbol et al. (2017) based on the third derivative of the ICG signal.

    This algorithm extracts B-points based on the maximum of the third derivative of the ICG signal within a 300ms
    window before the C-point.

    For more information, see [Arb17]_.

    References
    ----------
    .. [Arb17] Árbol, J. R., Perakakis, P., Garrido, A., Mata, J. L., Fernández-Santaella, M. C., & Vila, J. (2017).
        Mathematical detection of aortic valve opening (B-point) in impedance cardiography: A comparison of three
        popular algorithms. Psychophysiology, 54(3), 350-357. https://doi.org/10.1111/psyp.12799

    """

    # input parameters
    search_window_start_ms: Parameter[str | int]  # either 'R' or integer defining window length in ms
    correct_outliers: Parameter[bool]

    def __init__(
        self,
        search_window_start_ms: str | int | None = 300,
        handle_missing_events: HANDLE_MISSING_EVENTS = "warn",
    ):
        """Initialize new ``BPointExtractionArbol2017ThirdDerivative`` algorithm instance.

        Parameters
        ----------
        search_window_start_ms : int or str, optional
           Start of the window in which the algorithm searches for the B-point, relative to the C-point. Can be one of:
                * 'R' -> search B-point in the region between R-peak and C-point
                * int -> search B-point in the region between xx ms before C-point and C-point
                    (300 ms -> see Arbol 2017, 3rd derivative-based algorithm)
        handle_missing_events : one of {"warn", "raise", "ignore"}, optional
            How to handle failing event extraction. Can be one of:
                * "warn": issue a warning and set the event to NaN
                * "raise": raise an ``EventExtractionError``
                * "ignore": ignore the error and continue with the next event
            Default: "warn"


        """
        super().__init__(handle_missing_events=handle_missing_events)
        self.search_window_start_ms = search_window_start_ms

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

        This algorithm extracts B-points based on the maximum of the third derivative of the ICG signal within a 300ms
        window before the C-point.

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

        # result dfs
        b_points = pd.DataFrame(index=heartbeats.index, columns=["b_point_sample", "nan_reason"])

        # used subsequently to store ids of heartbeats where no B was detected because there was no C
        # (Bs should always be found, since they are set to the max of the 3rd derivative, and there is always a max)
        heartbeats_no_c_b = []
        # (but in case of wrongly detected Cs, the search window might be invalid, then no B can be found)
        heartbeats_no_b = []

        icg_2nd_der = np.gradient(icg)
        icg_3rd_der = np.gradient(icg_2nd_der)

        # search B-point for each heartbeat of the given signal
        for idx, data in heartbeats.iterrows():
            # calculate R-peak and C-point position relative to start of current heartbeat
            heartbeat_r_peak = data["r_peak_sample"]
            heartbeat_c_point = c_points.loc[idx, "c_point_sample"]

            # C-point can be NaN, then, extraction of B is not possible, so B is set to NaN
            if pd.isna(heartbeat_c_point):
                heartbeats_no_c_b.append(idx)
                b_points.loc[idx, "b_point_sample"] = np.nan
                b_points.loc[idx, "nan_reason"] = "c_point_nan"
                continue

            # set window end to C-point position and set window start according to specified method
            window_end = heartbeat_c_point
            if self.search_window_start_ms == "R":
                window_start = heartbeat_r_peak
            elif isinstance(self.search_window_start_ms, int):
                window_length_samples = int((self.search_window_start_ms / 1000) * sampling_rate_hz)
                window_start = heartbeat_c_point - window_length_samples
            else:
                raise AttributeError("Wrong value for 'window_b_detection_ms'. Must be 'R' or int.")

            # might happen for wrongly detected Cs (search window becomes invalid)
            if window_start < 0 or window_end < 0:
                heartbeats_no_b.append(idx)
                b_points.loc[idx, "b_point_sample"] = np.nan
                b_points.loc[idx, "nan_reason"] = "invalid_b_point_search_window"
                continue

            # find max in B window and calculate B-point relative to signal start
            b_window = icg_3rd_der[window_start:window_end]
            b_window_max = np.argmax(b_window)
            b_point = b_window_max + window_start
            b_points.loc[idx, "b_point_sample"] = b_point

        # inform user about missing B-points
        if len(heartbeats_no_c_b) > 0 or len(heartbeats_no_b) > 0:
            nan_rows = b_points[b_points["b_point_sample"].isna()]
            n = len(nan_rows)
            nan_rows = nan_rows.drop(index=heartbeats_no_c_b)
            nan_rows = nan_rows.drop(index=heartbeats_no_b)

            missing_str = (
                f"No B-point detected in {n} heartbeats:\n"
                f"- For heartbeats {heartbeats_no_c_b} no B point could be extracted, "
                f"because there was no C point\n"
                f"- For heartbeats {heartbeats_no_b} the search window was invalid probably due to "
                f"wrongly detected C points\n"
                f"- for heartbeats {nan_rows.index.to_numpy()} apparently also no B point was found "
                f"for some other reasons"
            )

            if self.handle_missing_events == "warn":
                warnings.warn(missing_str)
            elif self.handle_missing_events == "raise":
                raise EventExtractionError(missing_str)

        b_points = b_points.astype({"b_point_sample": "Int64", "nan_reason": "object"})
        is_b_point_dataframe(b_points)

        self.points_ = b_points
        return self
