import warnings
from typing import Optional

import numpy as np
import pandas as pd
from biopsykit.signals._base_extraction import BaseExtraction, EXTRACTION_HANDLING_BEHAVIOR
from scipy.signal import argrelextrema, argrelmin
from tpcp import Parameter, make_action_safe

from biopsykit.utils._datatype_validation_helper import _assert_is_dtype, _assert_has_columns
from biopsykit.utils.exceptions import EventExtractionError


class BPointExtractionForouzanfar2019(BaseExtraction):
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
        signal_clean: pd.DataFrame,
        heartbeats: pd.DataFrame,
        c_points: pd.DataFrame,
        sampling_rate_hz: int,
        *,
        handle_missing: Optional[EXTRACTION_HANDLING_BEHAVIOR] = "warn",
    ):
        """Function which extracts B-points from given ICG cleaned signal.

        Args:
            signal_clean:
                cleaned ICG signal
            heartbeats:
                pd.DataFrame containing one row per segmented heartbeat, each row contains start, end, and R-peak
                location (in samples from beginning of signal) of that heartbeat, index functions as id of heartbeat
            c_points:
                pd.DataFrame containing one row per segmented C-point, each row contains location
                (in samples from beginning of signal) of that C-point or NaN if the location of that C-point
                is not correct
            sampling_rate_hz:
                sampling rate of ECG signal in hz

        Returns
        -------
            saves resulting B-point locations (samples) in points_ attribute of super class,
            index is C-point (/heartbeat) id
        """
        # sanitize input signal
        signal_clean = signal_clean.squeeze()

        # Create the B-Point/A-Point Dataframes with the index of the heartbeat_list
        b_points = pd.DataFrame(index=heartbeats.index, columns=["b_point_sample"])

        # check whether the c_points contain NaN
        check_c_points = np.isnan(c_points.values.astype(float))

        # Calculate the second- and third-derivative of the ICG-signal
        second_der = np.gradient(signal_clean)
        third_der = np.gradient(second_der)

        # print(c_points)

        for idx, data in heartbeats[1:].iterrows():
            # check if c_points contain NaN. If this is the case, set the b_point to NaN
            if check_c_points[idx] | check_c_points[idx - 1]:
                b_points["b_point_sample"].iloc[idx] = np.NaN
                continue
            else:
                # Detect the main peak in the dZ/dt signal (C-Point)
                c_point = c_points["c_point_sample"].iloc[idx]

            # Compute the beat to beat interval
            beat_to_beat = c_points["c_point_sample"].iloc[idx] - c_points["c_point_sample"].iloc[idx - 1]

            # Compute the search interval for the A-Point
            search_interval = int(beat_to_beat / 3)

            # Detect the local minimum (A-Point) within one third of the beat to beat interval prior to the C-Point
            a_point = self.get_a_point(signal_clean, search_interval, c_point) + (c_point - search_interval)

            # Select the signal_segment between the A-Point and the C-Point
            signal_clean_segment = signal_clean.iloc[a_point : c_point + 1]

            # Define the C-Point amplitude which is used as a constraint for monotonic segment detection
            c_amplitude = signal_clean.iloc[c_point]

            # Step 4.1: Get the most prominent monotonic increasing segment between the A-Point and the C-Point
            start_sample, end_sample = (
                self.get_monotonic_increasing_segments_2nd_der(
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
                    b_points["b_point_sample"].iloc[idx] = np.NaN
                continue

            # Get the first third of the monotonic increasing segment
            start = start_sample
            end = end_sample - int((2 / 3) * (end_sample - start_sample))

            # 2nd derivative of the segment
            monotonic_segment_2nd_der = pd.DataFrame(second_der[start:end], columns=["2nd_der"])
            # 3rd derivative of the segment
            monotonic_segment_3rd_der = pd.DataFrame(third_der[start:end], columns=["3rd_der"])

            # Calculate the amplitude difference between the C-Point and the A-Point
            height = signal_clean.iloc[c_point] - signal_clean.iloc[a_point]

            # Compute the significant zero_crossings
            significant_zero_crossings = self.get_zero_crossings(
                monotonic_segment_3rd_der, monotonic_segment_2nd_der, height, sampling_rate_hz
            )

            # Compute the significant local maximums of the 3rd derivative of the most prominent monotonic segment
            significant_local_maximums = self.get_local_maximums(monotonic_segment_3rd_der, height, sampling_rate_hz)

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
        b_points.iloc[0] = b_points["b_point_sample"].iloc[1] - b_points["b_point_sample"].diff().iloc[2]

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
        _assert_has_columns(b_points, [["b_point_sample"]])

        self.points_ = b_points
        return self

    @staticmethod
    def get_a_point(signal_clean: pd.DataFrame, search_interval: int, c_point: int):
        signal_interval = signal_clean.iloc[(c_point - search_interval) : c_point]
        signal_minima = argrelmin(signal_interval.values, mode="wrap")
        a_point_idx = np.argmin(signal_interval.iloc[signal_minima[0]])
        a_point = signal_minima[0][a_point_idx]
        return a_point

    @staticmethod
    def get_monotonic_increasing_segments_2nd_der(
        signal_clean_segment: pd.DataFrame, second_der_segment: pd.DataFrame, height: int
    ):
        signal_clean_segment.index = np.arange(0, len(signal_clean_segment))
        monotony_df = pd.DataFrame(signal_clean_segment.values, columns=["icg"])
        monotony_df["2nd_der"] = second_der_segment
        monotony_df["borders"] = 0

        # C-Point is a possible end of the monotonic segment
        monotony_df["borders"].iat[-1] = "end_increase"
        # A-Point is a possible start of the monotonic segment
        monotony_df["borders"].iat[0] = "start_increase"

        # start_increase if the sign of the second derivative changes from negative to positive
        monotony_df.loc[
            ((monotony_df["2nd_der"][1:-2] < 0) & (monotony_df["2nd_der"].shift(-1) >= 0)), "borders"
        ] = "start_increase"
        # end_increase if the sign of the second derivative changes from positive to negative
        monotony_df.loc[
            ((monotony_df["2nd_der"][1:-2] >= 0) & (monotony_df["2nd_der"].shift(-1) < 0)), "borders"
        ] = "end_increase"

        # drop all samples that are no possible start-/ end-points
        monotony_df = monotony_df.drop(monotony_df[monotony_df["borders"] == 0].index)
        monotony_df = monotony_df.reset_index()
        # Drop start- and corresponding end-point, if their start value is higher than 1/2 of H
        monotony_df = monotony_df.drop(
            monotony_df[(monotony_df["borders"] == "start_increase") & (monotony_df["icg"] > int(height / 2))].index
            + 1,
            axis=0,
        )

        monotony_df = monotony_df.drop(
            monotony_df[(monotony_df["borders"] == "start_increase") & (monotony_df["icg"] > int(height / 2))].index,
            axis=0,
        )

        # Drop start- and corresponding end-point, if their end values does not reach at least 2/3 of H
        monotony_df = monotony_df.drop(
            monotony_df[(monotony_df["borders"] == "end_increase") & (monotony_df["icg"] < int(2 * height / 3))].index
            - 1,
            axis=0,
        )

        monotony_df = monotony_df.drop(
            monotony_df[(monotony_df["borders"] == "end_increase") & (monotony_df["icg"] < int(2 * height / 3))].index,
            axis=0,
        )

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
    def get_zero_crossings(
        monotonic_segment_3rd_der: pd.DataFrame,
        monotonic_segment_2nd_der: pd.DataFrame,
        height: int,
        sampling_rate_hz: int,
    ):
        constraint = 10 * height / sampling_rate_hz

        zero_crossings = np.where(np.diff(np.signbit(monotonic_segment_3rd_der["3rd_der"])))[0]
        zero_crossings = pd.DataFrame(zero_crossings, columns=["sample_position"])

        # Discard zero_crossings with negative to positive sign change
        significant_crossings = zero_crossings.drop(
            zero_crossings[monotonic_segment_2nd_der.iloc[zero_crossings["sample_position"]].values < 0].index, axis=0
        )

        # Discard zero crossings with slope higher than 10*H/f_s
        significant_crossings = significant_crossings.drop(
            significant_crossings[
                monotonic_segment_2nd_der.iloc[significant_crossings["sample_position"]].values >= constraint
            ].index,
            axis=0,
        )

        if isinstance(zero_crossings, type(None)):
            return pd.DataFrame([0], columns=["sample_position"])
        elif len(zero_crossings) == 0:
            return pd.DataFrame([0], columns=["sample_position"])
        else:
            return significant_crossings

    @staticmethod
    def get_local_maximums(monotonic_segment_3rd_der: pd.DataFrame, height: int, sampling_rate_hz: int):
        constraint = 4 * height / sampling_rate_hz

        local_maximums = argrelextrema(monotonic_segment_3rd_der["3rd_der"].values, np.greater_equal)[0]
        local_maximums = pd.DataFrame(local_maximums, columns=["sample_position"])

        significant_maximums = local_maximums.drop(
            local_maximums[
                monotonic_segment_3rd_der["3rd_der"].iloc[local_maximums["sample_position"]].values < constraint
            ].index,
            axis=0,
        )

        if isinstance(significant_maximums, type(None)):
            return pd.DataFrame([0], columns=["sample_position"])
        elif len(significant_maximums) == 0:
            return pd.DataFrame([0], columns=["sample_position"])
        else:
            # print(f"Received significant maximum!")
            return significant_maximums
