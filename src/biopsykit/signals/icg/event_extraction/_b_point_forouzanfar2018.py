import warnings
from typing import Optional

import numpy as np
import pandas as pd
from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS
from biopsykit.signals.icg.event_extraction._base_b_point_extraction import BaseBPointExtraction
from biopsykit.utils._datatype_validation_helper import _assert_has_columns, _assert_is_dtype
from biopsykit.utils.exceptions import EventExtractionError
from scipy.signal import argrelextrema, argrelmin
from tpcp import Parameter

__all__ = ["BPointExtractionForouzanfar2018"]


class BPointExtractionForouzanfar2018(BaseBPointExtraction):
    """algorithm to extract B-point based on [Forouzanfar et al., 2018, Psychophysiology]."""

    # input parameters
    correct_outliers: Parameter[bool]

    def __init__(
        self,
        correct_outliers: Optional[bool] = False,
    ):
        """Initialize new BPointExtractionForouzanfar algorithm instance.

        Parameters
        ----------
        correct_outliers : bool
            Indicates whether to perform outlier correction (True) or not (False)
        standard_amplitude : bool
            Indicates whether to use the amplitude of the C-Point or the amplitude difference between the C-Point and
            the A-Point as constraint to detect the monotonic segment
        """
        self.correct_outliers = correct_outliers

    # @make_action_safe
    def extract(
        self,
        *,
        icg: pd.Series,
        heartbeats: pd.DataFrame,
        c_points: pd.DataFrame,
        sampling_rate_hz: int,
        handle_missing: Optional[HANDLE_MISSING_EVENTS] = "warn",
    ):
        """Extract B-points from given ICG cleaned signal.

        The results are saved in the 'points_' attribute of the class instance.

        Parameters
        ----------
        icg : :class:`~pandas.DataFrame`
            cleaned ICG signal
        heartbeats : :class:`~pandas.DataFrame`
            DataFrame containing one row per segmented heartbeat, each row contains start, end, and R-peak location
            (in samples from beginning of signal) of that heartbeat, index functions as id of heartbeat
        c_points : :class:`~pandas.DataFrame`
            DataFrame containing one row per segmented C-point, each row contains location
            (in samples from beginning of signal) of that C-point or NaN if the location of that C-point
            is not correct.
        sampling_rate_hz : int
            sampling rate of ECG signal in hz
        handle_missing : one of {"warn", "raise", "ignore"}, optional
            How to handle missing data in the input dataframes. Default: "warn"

        Returns
        -------
        self

        Raises
        ------
        :exc:`~biopsykit.utils.exceptions.EventExtractionError`
            If the C-Point contains NaN values and handle_missing is set to "raise"

        """
        # sanitize input signal
        icg = icg.squeeze()

        # Create the B-Point/A-Point Dataframes with the index of the heartbeat_list
        b_points = pd.DataFrame(index=heartbeats.index, columns=["b_point_sample", "nan_reason"])

        # check whether the c_points contain NaN
        check_c_points = pd.isna(c_points["c_point_sample"]).to_numpy()

        # Calculate the second- and third-derivative of the ICG-signal
        second_der = np.gradient(icg)
        third_der = np.gradient(second_der)

        for idx, data in heartbeats[1:].iterrows():
            # check if the current or the previous C-Point contain NaN . If this is the case, set the b_point to NaN
            if check_c_points[idx] or check_c_points[idx - 1]:
                b_points["b_point_sample"].iloc[idx] = np.nan
                b_points["nan_reason"].iloc[idx] = "c_point_nan"
                continue

            # Detect the main peak in the dZ/dt signal (C-Point)
            c_point = c_points["c_point_sample"].iloc[idx]

            # Compute the beat to beat interval
            beat_to_beat = c_points["c_point_sample"].iloc[idx] - c_points["c_point_sample"].iloc[idx - 1]

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
                self._get_monotonic_increasing_segments_2nd_der(
                    signal_clean_segment, second_der[a_point : c_point + 1], c_amplitude
                )
                + a_point
            )
            if (start_sample == a_point) & (end_sample == a_point):
                # warnings.warn(f"Could not find a monotonic increasing segment for heartbeat {idx}! "
                #              f"The B-Point was set to NaN")
                if self.correct_outliers:
                    b_points["b_point_sample"].iloc[idx] = data["r_peak_sample"]
                else:
                    b_points["b_point_sample"].iloc[idx] = np.nan
                    b_points["nan_reason"].iloc[idx] = "no_monotonic_segment"
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
            significant_zero_crossings = self._get_zero_crossings(
                monotonic_segment_3rd_der, monotonic_segment_2nd_der, height, sampling_rate_hz
            )

            # Compute the significant local maximums of the 3rd derivative of the most prominent monotonic segment
            significant_local_maximums = self._get_local_maximums(monotonic_segment_3rd_der, height, sampling_rate_hz)

            # Label the last zero crossing/ local maximum as the B-Point
            # If there are no zero crossings or local maximums use the first Point of the segment as B-Point
            significant_features = pd.concat([significant_zero_crossings, significant_local_maximums], axis=0) + start
            b_point = significant_features.iloc[np.argmin(c_point - significant_features)][0]

            """
            if not self.correct_outliers:
                if b_point < data['r_peak_sample']:
                    b_points['b_point'].iloc[idx] = np.NaN
                    #warnings.warn(f"The detected B-point is located before the R-Peak at heartbeat {idx}!"
                    #              f" The B-point was set to NaN.")
                else:
                    b_points['b_point'].iloc[idx] = b_point
            else:
                b_points['b_point'].iloc[idx] = b_point
            """
            b_points["b_point_sample"].iloc[idx] = b_point

        # interpolate the first B-Point with the second B-Point since it is not possible to detect a B-Point
        # for the first heartbeat
        b_points["b_point_sample"].iloc[0] = (
            b_points["b_point_sample"].iloc[1] - b_points["b_point_sample"].diff().iloc[2]
        )

        idx_nan = b_points["b_point_sample"].isna()
        if idx_nan.sum() > 0:
            idx_nan = list(b_points.index[idx_nan])
            missing_str = (
                f"Either 'r_peak' or 'c_point' contains NaN at positions {idx_nan}! The B-points were set to NaN."
            )
            if handle_missing == "warn":
                warnings.warn(missing_str)
            elif handle_missing == "raise":
                raise EventExtractionError(missing_str)

        _assert_is_dtype(b_points, pd.DataFrame)
        _assert_has_columns(b_points, [["b_point_sample", "nan_reason"]])

        self.points_ = b_points.convert_dtypes(infer_objects=True)
        return self

    @staticmethod
    def _get_a_point(icg: pd.DataFrame, search_interval: int, c_point: int):
        signal_interval = icg.iloc[(c_point - search_interval) : c_point]
        signal_minima = argrelmin(signal_interval.values, mode="wrap")
        a_point_idx = np.argmin(signal_interval.iloc[signal_minima[0]])
        a_point = signal_minima[0][a_point_idx]
        return a_point

    @staticmethod
    def _get_monotonic_increasing_segments_2nd_der(
        icg_segment: pd.DataFrame, icg_second_der_segment: pd.DataFrame, height: int
    ):
        icg_second_der_segment = icg_second_der_segment.squeeze()
        icg_segment.index = np.arange(0, len(icg_segment))
        monotony_df = pd.DataFrame(icg_segment.values, columns=["icg"])
        monotony_df = monotony_df.assign(**{"2nd_der": icg_second_der_segment, "borders": 0})

        # A-Point is a possible start of the monotonic segment
        monotony_df.loc[monotony_df.index[0], "borders"] = "start_increase"
        # C-Point is a possible end of the monotonic segment
        monotony_df.loc[monotony_df.index[-1], "borders"] = "end_increase"

        neg_pos_change_idx = np.where(np.ediff1d(np.sign(monotony_df["2nd_der"])) > 0)[0]
        monotony_df.loc[monotony_df.index[neg_pos_change_idx], "borders"] = "start_increase"

        pos_neg_change_idx = np.where(np.ediff1d(np.sign(monotony_df["2nd_der"])) < 0)[0]
        monotony_df.loc[monotony_df.index[pos_neg_change_idx], "borders"] = "end_increase"

        # drop all samples that are no possible start-/ end-points
        monotony_df = monotony_df.drop(monotony_df[monotony_df["borders"] == 0].index)
        monotony_df = monotony_df.reset_index()
        # Drop start- and corresponding end-point if their start value is higher than 1/2 of H
        start_index_drop_rule_a = monotony_df[
            (monotony_df["borders"] == "start_increase") & (monotony_df["icg"] > int(height / 2))
        ].index
        monotony_df = monotony_df.drop(start_index_drop_rule_a, axis=0)
        monotony_df = monotony_df.drop(start_index_drop_rule_a + 1, axis=0)

        # Drop start- and corresponding end-point if their end values does not reach at least 2/3 of H
        end_index_drop_rule_b = monotony_df[
            (monotony_df["borders"] == "end_increase") & (monotony_df["icg"] < int(2 * height / 3))
        ].index
        monotony_df = monotony_df.drop(end_index_drop_rule_b, axis=0)
        monotony_df = monotony_df.drop(end_index_drop_rule_b - 1, axis=0)

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

    @staticmethod
    def _get_zero_crossings(
        monotonic_segment_3rd_der: pd.DataFrame,
        monotonic_segment_2nd_der: pd.DataFrame,
        height: int,
        sampling_rate_hz: int,
    ):
        constraint = float(10 * height / sampling_rate_hz)

        zero_crossings = np.where(np.diff(np.signbit(monotonic_segment_3rd_der["3rd_der"])))[0]
        zero_crossings = pd.DataFrame(zero_crossings, columns=["sample_position"])

        # Discard zero_crossings with negative to positive sign change
        significant_crossings = zero_crossings.drop(
            zero_crossings[monotonic_segment_2nd_der.iloc[zero_crossings["sample_position"]].to_numpy() < 0].index,
            axis=0,
        )

        # Discard zero crossings with slope higher than 10*H/f_s
        significant_crossings = significant_crossings.drop(
            significant_crossings[
                monotonic_segment_2nd_der.iloc[significant_crossings["sample_position"]].to_numpy() >= constraint
            ].index,
            axis=0,
        )

        if isinstance(zero_crossings, type(None)) or len(zero_crossings) == 0:
            return pd.DataFrame([0], columns=["sample_position"])
        return significant_crossings

    @staticmethod
    def _get_local_maximums(monotonic_segment_3rd_der: pd.DataFrame, height: int, sampling_rate_hz: int):
        constraint = float(4 * height / sampling_rate_hz)

        local_maximums = argrelextrema(monotonic_segment_3rd_der["3rd_der"].to_numpy(), np.greater_equal)[0]
        local_maximums = pd.DataFrame(local_maximums, columns=["sample_position"])

        significant_maximums = local_maximums.drop(
            local_maximums[
                monotonic_segment_3rd_der["3rd_der"].iloc[local_maximums["sample_position"]].to_numpy() < constraint
            ].index,
            axis=0,
        )

        if isinstance(significant_maximums, type(None)):
            return pd.DataFrame([0], columns=["sample_position"])
        if len(significant_maximums) == 0:
            return pd.DataFrame([0], columns=["sample_position"])
        # print(f"Received significant maximum!")
        return significant_maximums