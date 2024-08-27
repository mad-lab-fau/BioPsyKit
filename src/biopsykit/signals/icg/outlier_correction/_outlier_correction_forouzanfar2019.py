from typing import Optional

import pandas as pd
import numpy as np
import warnings

from scipy.signal import butter, sosfiltfilt
from scipy.stats import median_abs_deviation
from statsmodels.tools.sm_exceptions import ValueWarning

from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.arima.model import ARIMA

from biopsykit.signals._base_extraction import BaseExtraction, EXTRACTION_HANDLING_BEHAVIOR
from biopsykit.utils._datatype_validation_helper import _assert_has_columns, _assert_is_dtype

__all__ = ["OutlierCorrectionForouzanfar2019"]


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
        corrected_b_points = pd.DataFrame(index=b_points.index, columns=["b_point_sample"])

        # stationarize the B-Point time data
        stationary_data = self.stationarize_data(b_points, c_points, sampling_rate_hz)

        # detect outliers
        outliers = self.detect_outliers(stationary_data)

        # print(f"Detected {len(outliers)} outliers in correction cycle 1!")
        if len(outliers) == 0:
            _assert_is_dtype(corrected_b_points, pd.DataFrame)
            _assert_has_columns(corrected_b_points, [["b_point_sample"]])
            self.points_ = b_points
            return self

        # Perform the outlier correction until no more outliers are detected
        counter = 2
        while len(outliers) > 0:
            if counter > 30:
                break
            corrected_b_points = self.correct_outliers(
                b_points, c_points, stationary_data, outliers, stationary_data["baseline"], sampling_rate_hz
            )
            stationary_data = self.stationarize_data(corrected_b_points, c_points, sampling_rate_hz)
            outliers = self.detect_outliers(stationary_data)
            # print(f"Detected {len(outliers)} outliers in correction cycle {counter}!")
            counter += 1

        _assert_is_dtype(corrected_b_points, pd.DataFrame)
        _assert_has_columns(corrected_b_points, [["b_point_sample"]])
        # print(f"No more outliers got detected!")
        self.points_ = corrected_b_points
        return self

    @staticmethod
    def stationarize_data(b_points: pd.DataFrame, c_points: pd.DataFrame, sampling_rate_hz: int) -> pd.DataFrame:
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

    @staticmethod
    def detect_outliers(stationary_data: pd.DataFrame) -> pd.DataFrame:
        median = np.median(stationary_data["statio_data"])
        median_abs_dev = median_abs_deviation(stationary_data["statio_data"], axis=0, nan_policy="propagate")
        outliers = pd.DataFrame(index=stationary_data.index, columns=["outliers"])
        outliers["outliers"] = False
        outliers.loc[(stationary_data["statio_data"] - median) > (3 * median_abs_dev), "outliers"] = True
        return outliers[outliers["outliers"]]

    @staticmethod
    def correct_outliers(
        b_points_uncorrected: pd.DataFrame,
        c_points: pd.DataFrame,
        statio_data: pd.DataFrame,
        outliers: pd.DataFrame,
        baseline: pd.DataFrame,
        sampling_rate_hz: int,
    ) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ValueWarning)

            data = statio_data["statio_data"].to_frame()
            data.loc[outliers.index, "statio_data"] = np.NaN
            input_endog = data["statio_data"].astype(float).interpolate()
            # Select order of the froward model
            maxlag = 30

            sel = None
            while maxlag > 0:
                try:
                    sel = ar_select_order(input_endog, maxlag=maxlag, ic="aic")
                except ValueError:
                    maxlag -= 1
                    print(f"Maxlag reduced to {maxlag}!")
                    continue
                break

            order = sel.ar_lags
            if order is None:
                order = [0]

            # fit the forward model
            arima_mod = ARIMA(endog=input_endog, order=(order, 0, 0))
            arima_res = arima_mod.fit(method="burg")
            # reverse the input data to get the backward model
            reversed_input = input_endog[::-1]
            # Select order of the backward model
            b_sel = ar_select_order(reversed_input, maxlag=maxlag, ic="aic")
            b_order = b_sel.ar_lags
            if b_order is None:
                b_order = [0]

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
            result.loc[data.index, "b_point_sample"] = (
                (
                    c_points["c_point_sample"][c_points.index[data.index]]
                    - (data["statio_data"] + baseline) * sampling_rate_hz
                )
                .fillna(0)
                .astype(int)
            )
            result["b_point_sample"] = result["b_point_sample"].replace(0, np.nan)

        return result
