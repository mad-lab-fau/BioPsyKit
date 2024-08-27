from typing import Optional

import pandas as pd
import numpy as np

from scipy.signal import butter, filtfilt
from scipy.stats import median_abs_deviation

from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.arima.model import ARIMA

from biopsykit.signals._base_extraction import BaseExtraction, EXTRACTION_HANDLING_BEHAVIOR


class OutlierCorrectionForouzanfar2019(BaseExtraction):
    """algorithm to correct outliers based on [Forouzanfar et al., 2018, Psychophysiology]"""

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
        corrected_b_points = pd.DataFrame(index=b_points.index, columns=["b_point"])

        # stationarize the B-Point time data
        stationary_data = self.stationarize_data(b_points, c_points, sampling_rate_hz)

        # detect outliers
        outliers = self.detect_outliers(stationary_data)
        # print(f"Detected {len(outliers)} outliers in correction cycle 1!")

        # Perform the outlier correction until no more outliers are detected
        counter = 2
        while len(outliers) > 0:
            if counter > 30:
                break
            corrected_b_points = self.correct_outliers(
                b_points,
                c_points,
                stationary_data,
                outliers,
                stationary_data["baseline"],
            )
            stationary_data = self.stationarize_data(corrected_b_points, c_points, sampling_rate_hz)
            outliers = self.detect_outliers(stationary_data)
            # print(f"Detected {len(outliers)} outliers in correction cycle {counter}!")
            counter += 1

        # print(f"No more outliers got detected!")
        self.points_ = corrected_b_points
        return self

    @staticmethod
    def stationarize_data(b_points: pd.DataFrame, c_points: pd.DataFrame, sampling_rate_hz: int) -> pd.DataFrame:
        dist_to_C = ((c_points["c_point"] - b_points["b_point"]) / sampling_rate_hz).to_frame()
        dist_to_C.columns = ["dist_to_C"]
        dist_to_C["b_point"] = b_points["b_point"]
        dist_to_C = dist_to_C.dropna()
        b, a = butter(4, Wn=0.1, btype="lowpass", output="ba", fs=1)
        baseline = filtfilt(b, a, dist_to_C["dist_to_C"].values)
        statio_data = (dist_to_C["dist_to_C"] - baseline).to_frame()
        statio_data.columns = ["statio_data"]
        statio_data["b_point"] = dist_to_C["b_point"]
        statio_data["baseline"] = baseline
        return statio_data

    @staticmethod
    def detect_outliers(stationary_data: pd.DataFrame) -> pd.DataFrame:
        median = np.median(stationary_data["statio_data"])
        median_abs_dev = median_abs_deviation(stationary_data["statio_data"], axis=0, nan_policy="propagate")
        outliers = pd.DataFrame(index=stationary_data.index, columns=["outliers"])
        outliers["outliers"] = False
        outliers.loc[(stationary_data["statio_data"] - median) > (3 * median_abs_dev), "outliers"] = True
        return outliers[outliers["outliers"] == True]

    @staticmethod
    def correct_outliers(
        b_points_uncorrected: pd.DataFrame,
        c_points: pd.DataFrame,
        statio_data: pd.DataFrame,
        outliers: pd.DataFrame,
        baseline: pd.DataFrame,
    ) -> pd.DataFrame:
        data = statio_data["statio_data"].to_frame()
        data.loc[outliers.index, "statio_data"] = np.NaN
        input_endog = data["statio_data"].astype(float).interpolate()
        # Select order of the froward model
        sel = ar_select_order(input_endog, maxlag=30, ic="aic")
        order = sel.ar_lags
        # fit the forward model
        arima_mod = ARIMA(endog=input_endog, order=(order, 0, 0))
        arima_res = arima_mod.fit(method="burg")
        # reverse the input data to get the backward model
        reversed_input = input_endog[::-1]
        # Select order of the backward model
        b_sel = ar_select_order(reversed_input, maxlag=30, ic="aic")
        b_order = b_sel.ar_lags
        # Fit the backward model
        b_arima_mod = ARIMA(endog=reversed_input, order=(b_order, 0, 0))
        b_arima_res = b_arima_mod.fit(method="burg")
        # predict the outlier values
        for idx in outliers.index:
            forward_prediction = arima_res.predict(idx, idx)
            backward_prediction = b_arima_res.predict(len(reversed_input) - idx, len(reversed_input) - idx)
            prediction = np.average([forward_prediction, backward_prediction])
            data.loc[idx, "statio_data"] = prediction
        result = b_points_uncorrected.copy()
        result.loc[data.index, "b_point"] = (
            (c_points.c_point[c_points.index[data.index]] - (data["statio_data"] + baseline) * 1000)
        ).astype(int)
        return result
