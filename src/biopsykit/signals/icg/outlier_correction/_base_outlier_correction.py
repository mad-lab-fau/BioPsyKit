from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import median_abs_deviation
from tpcp import Algorithm

from biopsykit.signals._base_extraction import EXTRACTION_HANDLING_BEHAVIOR

__all__ = ["BaseOutlierCorrection"]


class BaseOutlierCorrection(Algorithm):

    _action_methods = "correct_outliers"

    points_: pd.DataFrame

    def correct_outlier(
        self,
        *,
        b_points: pd.DataFrame,
        c_points: pd.DataFrame,
        sampling_rate_hz: int,
        handle_missing: Optional[EXTRACTION_HANDLING_BEHAVIOR] = "warn",
    ):
        raise NotImplementedError("Method 'correct_outliers' must be implemented in a subclass!")

    @staticmethod
    def detect_b_point_outlier(stationary_data: pd.DataFrame) -> pd.DataFrame:
        median = np.median(stationary_data["statio_data"])
        median_abs_dev = median_abs_deviation(stationary_data["statio_data"], axis=0, nan_policy="propagate")
        outliers = pd.DataFrame(index=stationary_data.index, columns=["outliers"])
        outliers["outliers"] = False
        outliers.loc[(stationary_data["statio_data"] - median) > (3 * median_abs_dev), "outliers"] = True
        return outliers[outliers["outliers"]]

    @staticmethod
    def stationarize_b_points(b_points: pd.DataFrame, c_points: pd.DataFrame, sampling_rate_hz: int) -> pd.DataFrame:
        dist_to_c_point = ((c_points["c_point_sample"] - b_points["b_point_sample"]) / sampling_rate_hz).to_frame()
        dist_to_c_point.columns = ["dist_to_c_point_ms"]
        dist_to_c_point["b_point_sample"] = b_points["b_point_sample"]
        dist_to_c_point = dist_to_c_point.dropna()

        sos = butter(4, Wn=0.1, btype="lowpass", output="sos", fs=1)
        baseline = sosfiltfilt(sos, dist_to_c_point["dist_to_c_point_ms"].values)

        # TODO Check if necessary after changing to sos? If yes, find out how it needs to be changed
        # if len(dist_to_C["dist_to_C"].values) <= 3 * max(len(b), len(a)):
        #     last_row = dist_to_C.iloc[-1]
        #     num_rows_to_append = ((3 * max(len(b), len(a))) - len(dist_to_C)) + 1  # +1 to ensure it's enough
        #     additional_rows = pd.DataFrame([last_row] * num_rows_to_append, columns=dist_to_C.columns)
        #     dist_to_C = pd.concat([dist_to_C, additional_rows], ignore_index=True)
        # baseline = filtfilt(b, a, dist_to_C["dist_to_C"].values)
        # baseline = baseline[:length]
        # dist_to_C = dist_to_C[:length]

        statio_data = (dist_to_c_point["dist_to_c_point_ms"] - baseline).to_frame()
        statio_data.columns = ["statio_data"]
        statio_data["b_point_sample"] = dist_to_c_point["b_point_sample"]
        statio_data["baseline"] = baseline
        return statio_data
