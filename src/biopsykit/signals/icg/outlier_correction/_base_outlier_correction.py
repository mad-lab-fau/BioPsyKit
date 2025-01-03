from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import median_abs_deviation
from tpcp import Algorithm

__all__ = ["BaseOutlierCorrection"]


class BaseOutlierCorrection(Algorithm):

    _action_methods = "correct_outlier"

    points_: pd.DataFrame

    def correct_outlier(
        self,
        *,
        b_points: pd.DataFrame,
        c_points: Optional[pd.DataFrame],
        sampling_rate_hz: float,
        **kwargs,
    ):
        raise NotImplementedError("Method 'correct_outliers' must be implemented in a subclass!")

    @staticmethod
    def detect_b_point_outlier(stationary_data: pd.DataFrame) -> pd.DataFrame:
        median = np.nanmedian(stationary_data["statio_data"])
        median_abs_dev = median_abs_deviation(stationary_data["statio_data"], axis=0, nan_policy="propagate")
        outliers = pd.DataFrame(index=stationary_data.index, columns=["outliers"])
        outliers["outliers"] = False
        outliers.loc[(stationary_data["statio_data"] - median) > (3 * median_abs_dev), "outliers"] = True
        outliers.loc[stationary_data["statio_data"].isna(), "outliers"] = True

        return outliers[outliers["outliers"]]

    @staticmethod
    def stationarize_b_points(b_points: pd.DataFrame, c_points: pd.DataFrame, sampling_rate_hz: float) -> pd.DataFrame:
        b_point_sample = b_points["b_point_sample"].astype(float)
        c_point_sample = c_points["c_point_sample"].astype(float)
        dist_to_c_point = ((c_point_sample - b_point_sample) / sampling_rate_hz).to_frame()
        dist_to_c_point.columns = ["dist_to_c_point_ms"]
        dist_to_c_point["b_point_sample"] = b_point_sample
        dist_to_c_point = dist_to_c_point.interpolate().interpolate(method="ffill").interpolate(method="bfill")

        sos = butter(4, Wn=0.1, btype="lowpass", output="sos", fs=1)

        # padlen for sosfiltfilt according to scipy documentation
        padlen = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
        if len(dist_to_c_point) <= padlen:
            # signal needs to be at least as long as padlen => reduce padlen for shorter signals
            padlen = len(dist_to_c_point) - 1
        baseline = sosfiltfilt(sos, dist_to_c_point["dist_to_c_point_ms"].values, padlen=padlen)

        # TODO Check if necessary after changing to sos? If yes, find out how it needs to be changed
        #  => changed by changing padlen

        statio_data = (dist_to_c_point["dist_to_c_point_ms"] - baseline).to_frame()
        statio_data.columns = ["statio_data"]
        statio_data["b_point_sample"] = dist_to_c_point["b_point_sample"]
        statio_data["baseline"] = baseline
        statio_data = statio_data.assign(padlen=padlen)

        return statio_data
