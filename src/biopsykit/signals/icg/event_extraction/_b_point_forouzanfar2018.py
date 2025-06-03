import warnings

import numpy as np
import pandas as pd
from scipy.signal import argrelmin
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

__all__ = ["BPointExtractionForouzanfar2018"]


class BPointExtractionForouzanfar2018(BaseBPointExtraction, CanHandleMissingEventsMixin):
    """B-point extraction algorithm by Forouzanfar et al. (2018).

    This algorithm extracts B-points based on the detection of the most prominent monotonic increasing segment
    between the A-Point (the local minimum within one third of the beat-to-beat interval prior to the C-Point)
    and the C-Point. The B-Point is then detected as the last zero crossing or local maximum of the third derivative
    of the ICG signal within the segment.

    For more information, see [For18]_.

    References
    ----------
    .. [For18] Forouzanfar, M., Baker, F. C., De Zambotti, M., McCall, C., Giovangrandi, L., & Kovacs, G. T. A. (2018).
        Toward a better noninvasive assessment of preejection period: A novel automatic algorithm for B-point detection
        and correction on thoracic impedance cardiogram. Psychophysiology, 55(8), e13072.
        https://doi.org/10.1111/psyp.13072

    """

    # input parameters
    scaling_factor: Parameter[float]
    correct_outliers: Parameter[bool]

    def __init__(
        self,
        scaling_factor: float = 2000,
        correct_outliers: bool | None = False,
        handle_missing_events: HANDLE_MISSING_EVENTS = "warn",
    ):
        """Initialize new ``BPointExtractionForouzanfar2018`` instance.

        .. note:: In the original paper, the authors report the *sampling frequency* of the ICG signal as the scaling
        factor. Since this would change the algorithm behavior depending on the sampling rate of the ICG signal,
        this implementation introduces a scaling factor that can be set by the user. By default, the scaling factor is
        set to 2000 (corresponding to a sampling rate of the original data of 2000 Hz) instead of using the
        sampling rate of the ICG signal.

        Parameters
        ----------
        scaling_factor : float, optional
            Scaling factor for the B-point extraction algorithm. Default: 2000
        correct_outliers : bool, optional
            True to correct outliers, False to set B-Point to NaN if no monotonic segment is found. Default: False
        handle_missing_events : one of {"warn", "raise", "ignore"}, optional
            How to handle failing event extraction. Can be one of:
                * "warn": issue a warning and set the event to NaN
                * "raise": raise an ``EventExtractionError``
                * "ignore": ignore the error and continue with the next event
            Default: "warn"

        """
        super().__init__(handle_missing_events=handle_missing_events)
        self.scaling_factor = scaling_factor
        self.correct_outliers = correct_outliers

    # @make_action_safe
    def extract(  # noqa: PLR0915
        self,
        *,
        icg: IcgRawDataFrame,
        heartbeats: HeartbeatSegmentationDataFrame,
        c_points: CPointDataFrame,
        sampling_rate_hz: float | None,  # noqa: ARG002
    ):
        """Extract B-points from given ICG derivative signal.

        The algorithm extracts B-points based on the detection of the most prominent monotonic increasing segment
        between the A-Point (the local minimum within one third of the beat-to-beat interval prior to the C-Point)
        and the C-Point. The B-Point is then detected as the last zero crossing or local maximum of the third derivative
        of the ICG signal within the segment.

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
        # sanitize input signal
        self._check_valid_missing_handling()
        is_icg_raw_dataframe(icg)
        is_heartbeat_segmentation_dataframe(heartbeats)
        is_c_point_dataframe(c_points)
        icg = sanitize_input_dataframe_1d(icg, column="icg_der")
        icg = icg.squeeze()

        # Create the B-Point/A-Point Dataframes with the index of the heartbeat_list
        b_points = pd.DataFrame(index=heartbeats.index, columns=["b_point_sample", "nan_reason"])

        # get the c_point locations from the c_points dataframe
        c_points = c_points["c_point_sample"]

        # Calculate the second- and third-derivative of the ICG-signal
        second_der = np.gradient(icg)
        third_der = np.gradient(second_der)

        for idx, data in heartbeats.iloc[1:].iterrows():
            # Get the C-Point location at the current heartbeat id
            c_point = c_points[idx]
            prev_c_point = c_points[idx - 1]

            # check if the current or the previous C-Point contain NaN. If this is the case, set the b_point to NaN
            if pd.isna(c_point) or pd.isna(prev_c_point):
                b_points.loc[idx, "b_point_sample"] = np.nan
                b_points.loc[idx, "nan_reason"] = "c_point_nan"
                continue

            # Compute the beat to beat interval
            beat_to_beat = c_point - prev_c_point

            # Compute the search interval for the A-Point
            search_interval = int(beat_to_beat / 3)

            # Detect the local minimum (A-Point) within one third of the beat to beat interval prior to the C-Point
            a_point = self._get_a_point(icg, search_interval, c_point) + (c_point - search_interval)

            # Select the signal_segment between the A-Point and the C-Point
            signal_clean_segment = icg.iloc[a_point : c_point + 1]

            # Define the C-Point amplitude which is used as a constraint for monotonic segment detection
            c_amplitude = icg.iloc[c_point]

            # Step 4.1: Get the most prominent monotonic increasing segment between the A-Point and the C-Point
            start_sample, end_sample = (
                self._get_most_prominent_monotonic_increasing_segment(signal_clean_segment, c_amplitude) + a_point
            )

            if (start_sample == a_point) & (end_sample == a_point):
                if self.correct_outliers:
                    b_points.loc[idx, "b_point_sample"] = data["r_peak_sample"]
                else:
                    b_points.loc[idx, "b_point_sample"] = np.nan
                    b_points.loc[idx, "nan_reason"] = "no_monotonic_segment"
                continue

            # Get the first third of the monotonic increasing segment
            start = start_sample
            end = end_sample - int((2 / 3) * (end_sample - start_sample))

            # 2nd derivative of the segment
            monotonic_segment_2nd_der = pd.DataFrame(second_der[start:end], columns=["2nd_der"])
            # 3rd derivative of the segment
            monotonic_segment_3rd_der = pd.DataFrame(third_der[start:end], columns=["3rd_der"])

            # Calculate the amplitude difference between the C-Point and the A-Point
            height = icg.iloc[c_point] - icg.iloc[a_point]

            # Compute the significant zero_crossings
            significant_zero_crossings = self._get_zero_crossings_3rd_derivative(
                monotonic_segment_3rd_der, monotonic_segment_2nd_der, height
            )

            # Compute the significant local maxima of the 3rd derivative of the most prominent monotonic segment
            significant_local_maxima = self._get_local_maxima_3rd_derivative(monotonic_segment_3rd_der, height)

            # Label the last zero crossing/ local maximum as the B-Point
            # If there are no zero crossings or local maxima use the first Point of the segment as B-Point
            significant_features = pd.concat([significant_zero_crossings, significant_local_maxima], axis=0) + start
            if len(significant_features) == 0:
                b_point = start
            else:
                b_point = significant_features.iloc[np.argmin(c_point - significant_features)].iloc[0]

            b_points.loc[idx, "b_point_sample"] = b_point

        # interpolate the first B-Point with the second B-Point since it is not possible to detect a B-Point
        # for the first heartbeat
        b_points.loc[b_points.index[0], "b_point_sample"] = (
            b_points.loc[b_points.index[1], "b_point_sample"] - b_points["b_point_sample"].diff().loc[b_points.index[2]]
        )

        idx_nan = b_points["b_point_sample"].isna()
        if idx_nan.sum() > 0:
            idx_nan = list(b_points.index[idx_nan])
            missing_str = (
                f"Either 'r_peak' or 'c_point' contains NaN at positions {idx_nan}! The B-points were set to NaN."
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
    def _get_a_point(icg: pd.Series, search_interval: int, c_point: int):
        signal_interval = icg.iloc[(c_point - search_interval) : c_point]
        signal_minima = argrelmin(signal_interval.values, mode="wrap")

        a_point_candidates = signal_interval.iloc[signal_minima[0]]
        if len(a_point_candidates) == 0:
            # no local minima found => return the argmin of the interval
            return np.argmin(signal_interval)

        a_point_idx = np.argmin(a_point_candidates)
        a_point = signal_minima[0][a_point_idx]
        return a_point

    @staticmethod
    def _get_most_prominent_monotonic_increasing_segment(icg_segment: pd.Series, height: int):
        icg_segment = icg_segment.copy()
        icg_2nd_der_segment = np.gradient(icg_segment)
        icg_segment.index = np.arange(0, len(icg_segment))
        monotony_df = pd.DataFrame(icg_segment.values, columns=["icg"])
        monotony_df = monotony_df.assign(**{"2nd_der": icg_2nd_der_segment, "borders": pd.NA})

        # A-Point is a possible start of the monotonic segment
        monotony_df.loc[monotony_df.index[0], "borders"] = "start_increase"
        # C-Point is a possible end of the monotonic segment
        monotony_df.loc[monotony_df.index[-1], "borders"] = "end_increase"

        neg_pos_change_idx = np.where(np.diff(np.sign(monotony_df["2nd_der"])) > 0)[0]
        monotony_df.loc[monotony_df.index[neg_pos_change_idx], "borders"] = "start_increase"

        pos_neg_change_idx = np.where(np.diff(np.sign(monotony_df["2nd_der"])) < 0)[0]
        monotony_df.loc[monotony_df.index[pos_neg_change_idx], "borders"] = "end_increase"

        # drop all samples that are no possible start-/ end-points
        monotony_df = monotony_df.drop(monotony_df[pd.isna(monotony_df["borders"])].index)
        monotony_df = monotony_df.reset_index()

        # Drop start- and corresponding end-point if their start value is higher than 1/2 of H
        start_index_drop_rule_a = monotony_df[
            (monotony_df["borders"] == "start_increase") & (monotony_df["icg"] > height / 2)
        ].index
        start_index_drop_rule_a = start_index_drop_rule_a.union(start_index_drop_rule_a + 1)
        monotony_df = monotony_df.drop(index=start_index_drop_rule_a)

        # Drop start- and corresponding end-point if their end values does not reach at least 2/3 of H
        end_index_drop_rule_b = monotony_df[
            (monotony_df["borders"] == "end_increase") & (monotony_df["icg"] < 2 * height / 3)
        ].index

        end_index_drop_rule_b = end_index_drop_rule_b.union(end_index_drop_rule_b - 1)
        monotony_df = monotony_df.drop(index=monotony_df.iloc[end_index_drop_rule_b].index)

        # Select the monotonic segment with the highest amplitude difference
        start_sample = 0
        end_sample = 0
        if len(monotony_df) > 2:
            idx = np.argmax(monotony_df["icg"].diff())
            start_sample = monotony_df["index"].iloc[idx - 1]
            end_sample = monotony_df["index"].iloc[idx]
        elif len(monotony_df) != 0:
            start_sample = monotony_df["index"].iloc[0]
            end_sample = monotony_df["index"].iloc[-1]

        return start_sample, end_sample  # That are not absolute positions yet

    def _get_zero_crossings_3rd_derivative(
        self, monotonic_segment_3rd_der: pd.DataFrame, monotonic_segment_2nd_der: pd.DataFrame, height: int
    ):
        constraint = float(10 * height / self.scaling_factor)

        zero_crossings = np.where(np.diff(np.signbit(monotonic_segment_3rd_der["3rd_der"])))[0]
        zero_crossings = pd.DataFrame(zero_crossings, columns=["sample_position"])

        # Discard zero_crossings with negative to positive sign change
        significant_crossings = zero_crossings.drop(
            zero_crossings[monotonic_segment_2nd_der.iloc[zero_crossings["sample_position"]].to_numpy() < 0].index,
            axis=0,
        )

        # Discard zero crossings with slope higher than 10*H/scaling_factor
        significant_crossings = significant_crossings.drop(
            significant_crossings[
                monotonic_segment_2nd_der.iloc[significant_crossings["sample_position"]].to_numpy() >= constraint
            ].index,
            axis=0,
        )

        if isinstance(zero_crossings, type(None)) or len(zero_crossings) == 0:
            return pd.DataFrame([0], columns=["sample_position"])
        return significant_crossings

    def _get_local_maxima_3rd_derivative(self, monotonic_segment_3rd_der: pd.DataFrame, height: int):
        constraint = float(4 * height / self.scaling_factor)
        # compute gradient
        monotonic_segment_3rd_der_gradient = np.gradient(monotonic_segment_3rd_der.squeeze())

        # find zero-crossings of the gradient to determine local maxima
        local_maxima = np.where(np.diff(np.sign(monotonic_segment_3rd_der_gradient)) < 0)[0]
        # add 1 to the index to get the correct position
        local_maxima += 1
        local_maxima = pd.DataFrame(local_maxima, columns=["sample_position"])

        significant_maxima = local_maxima.drop(
            local_maxima[
                monotonic_segment_3rd_der["3rd_der"].iloc[local_maxima["sample_position"]].to_numpy() < constraint
            ].index,
            axis=0,
        )

        if isinstance(significant_maxima, type(None)):
            return pd.DataFrame([0], columns=["sample_position"])
        if len(significant_maxima) == 0:
            return pd.DataFrame([0], columns=["sample_position"])
        return significant_maxima
