from typing import Optional

import pandas as pd
import numpy as np
import warnings

from statsmodels.tools.sm_exceptions import ValueWarning

from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.arima.model import ARIMA

from biopsykit.signals._base_extraction import EXTRACTION_HANDLING_BEHAVIOR
from biopsykit.signals.icg.outlier_correction._base_outlier_correction import BaseOutlierCorrection
from biopsykit.utils._datatype_validation_helper import _assert_has_columns, _assert_is_dtype

__all__ = ["OutlierCorrectionForouzanfar2019"]


# TODO add verbosity option


class OutlierCorrectionForouzanfar2019(BaseOutlierCorrection):
    """algorithm to correct outliers based on [Forouzanfar et al., 2018, Psychophysiology]"""

    # @make_action_safe
    def correct_outlier(
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
        stationary_data = self.stationarize_b_points(b_points, c_points, sampling_rate_hz)

        # detect outliers
        outliers = self.detect_b_point_outlier(stationary_data)

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
            corrected_b_points = self._correct_outlier_autoregression(
                b_points, c_points, stationary_data, outliers, stationary_data["baseline"], sampling_rate_hz
            )
            stationary_data = self.stationarize_b_points(corrected_b_points, c_points, sampling_rate_hz)
            outliers = self.detect_b_point_outlier(stationary_data)
            # print(f"Detected {len(outliers)} outliers in correction cycle {counter}!")
            counter += 1

        _assert_is_dtype(corrected_b_points, pd.DataFrame)
        _assert_has_columns(corrected_b_points, [["b_point_sample"]])
        # print(f"No more outliers got detected!")

        self.points_ = corrected_b_points.convert_dtypes(infer_objects=True)
        return self

    @staticmethod
    def _correct_outlier_autoregression(
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
