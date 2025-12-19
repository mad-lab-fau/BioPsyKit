import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.arima.model import ARIMA

from biopsykit.signals.icg.outlier_correction._base_outlier_correction import BaseBPointOutlierCorrection

__all__ = ["OutlierCorrectionForouzanfar2018"]


from biopsykit.utils.dtypes import BPointDataFrame, CPointDataFrame, is_b_point_dataframe, is_c_point_dataframe

# TODO add verbosity option


class OutlierCorrectionForouzanfar2018(BaseBPointOutlierCorrection):
    """B-Point outlier correction algorithm based on Forouzanfar et al. (2018).

    This algorithm corrects outliers in B-Point data using an autoregressive model.

    For more information, see [For18]_.

    References
    ----------
    .. [For18] Forouzanfar, M., Baker, F. C., De Zambotti, M., McCall, C., Giovangrandi, L., & Kovacs, G. T. A. (2018).
        Toward a better noninvasive assessment of preejection period: A novel automatic algorithm for B-point detection
        and correction on thoracic impedance cardiogram. Psychophysiology, 55(8), e13072.
        https://doi.org/10.1111/psyp.13072

    """

    # @make_action_safe
    def correct_outlier(
        self,
        *,
        b_points: BPointDataFrame,
        c_points: CPointDataFrame,
        sampling_rate_hz: float,
        **kwargs,
    ):
        """Perform outlier correction on B-Point data.

        The outliers are corrected using an autoregressive model.

        The results of the outlier correction are saved in the ``points_`` attribute of the super class.

        Parameters
        ----------
        b_points : :class:`~pandas.DataFrame`
            Extracted B-points. Each row contains the B-point location (in samples from beginning of signal) for each
            heartbeat, index functions as id of heartbeat. B-point locations can be NaN if no B-points were detected
            for certain heartbeats.
        c_points : :class:`~pandas.DataFrame`
            Extracted C-points. Each row contains the C-point location (in samples from beginning of signal) for each
            heartbeat, index functions as id of heartbeat. C-point locations can be NaN if no C-points were detected
            for certain heartbeats.
        sampling_rate_hz : int
            sampling rate of ICG derivative signal in hz
        kwargs : dict
            Additional keyword arguments:
                * verbose: bool, optional
                    Whether to print additional information. Default: False

        Returns
        -------
            self

        """
        verbose = kwargs.get("verbose", False)
        is_b_point_dataframe(b_points)
        is_c_point_dataframe(c_points)

        corrected_b_points = pd.DataFrame(index=b_points.index, columns=["b_point_sample"])

        # stationarize the B-Point time data
        stationary_data = self.stationarize_b_points(b_points, c_points, sampling_rate_hz)
        b_points_nan = b_points.loc[b_points["b_point_sample"].isna()]
        stationary_data.loc[b_points_nan.index, "statio_data"] = np.nan

        # detect outliers
        outliers = self.detect_b_point_outlier(stationary_data)

        counter = 1
        if verbose:
            print(f"Detected {len(outliers)} outliers in correction cycle {counter}!")
        if len(outliers) == 0:
            is_b_point_dataframe(b_points)
            self.points_ = b_points
            return self

        # Perform the outlier correction until no more outliers are detected
        while len(outliers) > 0:
            if counter >= 30:
                break
            corrected_b_points = self._correct_outlier_autoregression(
                b_points, c_points, stationary_data, outliers, stationary_data["baseline"], sampling_rate_hz, **kwargs
            )
            stationary_data = self.stationarize_b_points(corrected_b_points, c_points, sampling_rate_hz)
            outliers = self.detect_b_point_outlier(stationary_data)
            if verbose:
                print(f"Detected {len(outliers)} outliers in correction cycle {counter}!")
            counter += 1

        if verbose:
            print("No more outliers got detected!")
        corrected_b_points = corrected_b_points.astype({"b_point_sample": "Int64", "nan_reason": "object"})
        is_b_point_dataframe(corrected_b_points)

        self.points_ = corrected_b_points
        return self

    @staticmethod
    def _correct_outlier_autoregression(
        b_points_uncorrected: pd.DataFrame,
        c_points: pd.DataFrame,
        statio_data: pd.DataFrame,
        outliers: pd.DataFrame,
        baseline: pd.DataFrame,
        sampling_rate_hz: int,
        **kwargs,
    ) -> pd.DataFrame:
        """Correct outliers in B-Point data using an autoregressive model.

        Parameters
        ----------
        b_points_uncorrected: :class:`~pandas.DataFrame`
            Dataframe containing the extracted B-Points per heartbeat, index functions as id of heartbeat
        c_points: :class:`~pandas.DataFrame`
            Dataframe containing the extracted C-Points per heartbeat, index functions as id of heartbeat
        statio_data: :class:`~pandas.DataFrame`
            Stationarized B-Point data
        outliers: :class:`~pandas.DataFrame`
            Dataframe containing the detected outliers
        baseline: :class:`~pandas.DataFrame`
            Baseline of the stationarized B-Point data
        sampling_rate_hz: int
            Sampling rate of ICG signal in hz
        **kwargs
            Additional keyword arguments:
                * verbose: bool, optional
                    Whether to print additional information. Default: False

        Returns
        -------
        :class:`~pandas.DataFrame`
            Corrected B-Point data

        """
        verbose = kwargs.get("verbose", False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ValueWarning)
            warnings.simplefilter("ignore", category=FutureWarning)

            data = statio_data["statio_data"].to_frame()
            data.loc[outliers.index, "statio_data"] = np.nan
            input_endog = (
                data["statio_data"]
                .astype(float)
                .interpolate(method="linear")
                .interpolate(method="ffill")  # ensure that no NaN values are at the beginning
                .interpolate(method="bfill")  # ensure that no NaN values are at the end
            )

            # Select order of the froward model
            maxlag = 30

            sel = None
            while maxlag > 0:
                try:
                    sel = ar_select_order(input_endog, maxlag=maxlag, ic="aic")
                except ValueError:
                    maxlag -= 1
                    if verbose:
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
                backward_idx = (len(reversed_input) - idx) % len(reversed_input)
                backward_prediction = b_arima_res.predict(backward_idx, backward_idx)
                prediction = np.average([forward_prediction, backward_prediction])
                data.loc[idx, "statio_data"] = prediction
            result = b_points_uncorrected.copy()
            result.loc[data.index, "b_point_sample"] = (
                (
                    c_points["c_point_sample"][c_points.loc[data.index].index]
                    - (data["statio_data"] + baseline) * sampling_rate_hz
                )
                .fillna(0)
                .astype(int)
            )
            result["b_point_sample"] = result["b_point_sample"].replace(0, np.nan)

            result = result.assign(maxlag=maxlag, padlen=statio_data["padlen"].iloc[0])

        return result
