from typing import Optional

import pandas as pd
import numpy as np

from scipy.signal import butter, sosfiltfilt
from scipy.stats import median_abs_deviation

from biopsykit.signals._base_extraction import EXTRACTION_HANDLING_BEHAVIOR, BaseExtraction

__all__ = ["OutlierCorrectionInterpolation"]

from biopsykit.utils._datatype_validation_helper import _assert_is_dtype, _assert_has_columns


class OutlierCorrectionInterpolation(BaseExtraction):
    """algorithm to correct outliers based on Linear Interpolation"""

    # @make_action_safe
    def extract(
        self,
        *,
        b_points: pd.DataFrame,
        c_points: pd.DataFrame,
        sampling_rate_hz: int,
        handle_missing: Optional[EXTRACTION_HANDLING_BEHAVIOR] = "warn",
    ):
        """function which corrects outliers of given B-Point dataframe

        Args:
            b_points:
                pd.DataFrame containing the extracted B-Points per heartbeat, index functions as id of heartbeat
            c-points:
                pd.DataFrame containing the extracted C-Points per heartbeat, index functions as id of heartbeat
            sampling_rate_hz:
                sampling rate of ICG signal in hz

        Returns:
            saves resulting corrected B-point locations (samples) in points_ attribute of super class,
            index is B-point (/heartbeat) id
        """
        corrected_b_points = pd.DataFrame(index=b_points.index, columns=["b_point_sample"])

        # stationarize the B-Point time data
        stationary_data = self.stationarize_data(b_points, c_points, sampling_rate_hz)

        # detect outliers
        outliers = self.detect_outliers(stationary_data)
        print(f"Detected {len(outliers)} outliers in correction cycle 1!")
        if len(outliers) == 0:
            _assert_is_dtype(corrected_b_points, pd.DataFrame)
            _assert_has_columns(corrected_b_points, [["b_point_sample"]])
            return self

        # Perform the outlier correction until no more outliers are detected
        counter = 2
        while len(outliers) > 0:
            if counter > 200:
                break
            corrected_b_points = self.correct_linear(
                b_points, c_points, stationary_data, outliers, stationary_data["baseline"], sampling_rate_hz
            )
            # print(corrected_b_points)
            stationary_data = self.stationarize_data(corrected_b_points, c_points, sampling_rate_hz)
            outliers = self.detect_outliers(stationary_data)
            print(f"Detected {len(outliers)} outliers in correction cycle {counter}!")
            counter += 1

        print(f"No more outliers got detected!")

        _assert_is_dtype(corrected_b_points, pd.DataFrame)
        _assert_has_columns(corrected_b_points, [["b_point_sample"]])

        self.points_ = corrected_b_points
        return self

    @staticmethod
    def stationarize_data(b_points: pd.DataFrame, c_points: pd.DataFrame, sampling_rate_hz: int) -> pd.DataFrame:
        dist_to_c_point = ((c_points["c_point_sample"] - b_points["b_point_sample"]) / sampling_rate_hz).to_frame()
        dist_to_c_point.columns = ["dist_to_c_point_ms"]
        dist_to_c_point["b_point_sample"] = b_points["b_point_sample"]
        dist_to_c_point = dist_to_c_point.dropna()

        sos = butter(4, Wn=0.1, btype="lowpass", output="sos", fs=1)

        # TODO Check if necessary after changing to sos? If yes, find out how it needs to be changed
        # length = len(b_points)
        # if len(dist_to_c_point["dist_to_c_point_ms"].values) <= 3 * max(len(b), len(a)):
        #     last_row = dist_to_c_point.iloc[-1]
        #     num_rows_to_append = ((3 * max(len(b), len(a))) - len(dist_to_C)) + 1  # +1 to ensure it's enough
        #     additional_rows = pd.DataFrame([last_row] * num_rows_to_append, columns=dist_to_C.columns)
        #     dist_to_C = pd.concat([dist_to_C, additional_rows], ignore_index=True)
        # baseline = filtfilt(b, a, dist_to_c_point["dist_to_c_point_ms"].values)
        # baseline = baseline[:length]
        # dist_to_c_point = dist_to_c_point[:length]

        baseline = sosfiltfilt(sos, dist_to_c_point["dist_to_c_point_ms"].values)

        statio_data = (dist_to_c_point["dist_to_c_point_ms"] - baseline).to_frame()
        statio_data.columns = ["statio_data"]
        statio_data["b_point_sample"] = dist_to_c_point["b_point_sample"]
        statio_data["baseline"] = baseline
        return statio_data

    @staticmethod
    def detect_outliers(stationary_data: pd.DataFrame) -> pd.DataFrame:
        median_time = np.median(stationary_data["statio_data"])
        median_abs_dev_time = median_abs_deviation(stationary_data["statio_data"], axis=0, nan_policy="propagate")
        outliers = pd.DataFrame(index=stationary_data.index, columns=["outliers"])
        outliers["outliers"] = False
        outliers.loc[(stationary_data["statio_data"] - median_time) > (3 * median_abs_dev_time), "outliers"] = True
        return outliers[outliers["outliers"]]

    @staticmethod
    def correct_linear(
        b_points_uncorrected: pd.DataFrame,
        c_points: pd.DataFrame,
        statio_data: pd.DataFrame,
        outliers: pd.DataFrame,
        baseline: pd.DataFrame,
        sampling_rate_hz: int,
    ) -> pd.DataFrame:
        data = statio_data["statio_data"].to_frame()
        # insert NaN at the heartbeat id of the outliers
        data.loc[outliers.index, "statio_data"] = np.NaN

        # interpolate the outlier positions using linear interpolation
        data_interpol = data["statio_data"].astype(float).interpolate()

        corrected_b_points = b_points_uncorrected.copy()
        # Add the baseline back to the interpolated values
        corrected_b_points.loc[data.index, "b_point_sample"] = (
            ((c_points["c_point_sample"][c_points.index[data.index]] - (data_interpol + baseline) * sampling_rate_hz))
            .fillna(0)
            .astype(int)
        )
        corrected_b_points["b_point_sample"] = corrected_b_points["b_point_sample"].replace(0, np.nan)
        return corrected_b_points
