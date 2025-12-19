from collections.abc import Sequence
from typing import Any, ClassVar

import neurokit2 as nk
import numpy as np
import pandas as pd
from tpcp import Algorithm

__all__ = ["RPeakOutlierCorrection"]


class RPeakOutlierCorrection(Algorithm):
    """Outlier correction algorithm for R-peaks.

    This algorithm is used to correct outliers in the R-peak data. It uses the detected outliers from the
    :class:`~biopsykit.signals.ecg.outlier_correction.RPeakOutlierDetection` algorithms and replaces them with
    interpolated or smoothed (moving averaged) values.

    Attributes
    ----------
    ecg_processed_ : :class:`~pandas.DataFrame`
        The processed ECG data with outliers corrected and interpolated instantaneous heart rate time-series.
    points_ : :class:`~pandas.DataFrame`
        The processed R-peak data with corrected outliers and heart rate re-calculated.

    """

    _action_methods = "correct_outlier"

    IMPUTATION_TYPES: ClassVar[Sequence[str]] = ["linear_interpolation", "moving_average"]
    imputation_type: str
    imputation_params: dict[str, Any]

    ecg_processed_: pd.DataFrame
    points_: pd.DataFrame

    def __init__(
        self, *, imputation_type: str = "linear_interpolation", imputation_params: dict[str, Any] | None = None
    ) -> None:
        """Initialize new ``RPeakOutlierCorrection`` algorithm instance.

        Parameters
        ----------
        imputation_type : str, optional
            The type of imputation to use. Options are:
                * "linear_interpolation" (default): Use linear interpolation to fill in missing values.
                * "moving_average": Use moving average to fill in missing values. The window size (centered) can be
                specified in the ``imputation_params`` dictionary with the key "window_size".
            Default: "linear_interpolation"
        imputation_params : dict, optional
            additional parameters for the imputation method. For "moving_average", the window size can be specified with
            the key "window_size". Default: None

        """
        self.imputation_type = imputation_type
        self.imputation_params = imputation_params
        super().__init__()

    def _check_imputation_type_valid(self):
        if self.imputation_params is None:
            self.imputation_params = {}
        if self.imputation_type not in self.IMPUTATION_TYPES:
            raise ValueError(
                f"Invalid imputation type '{self.imputation_type}'. Valid options are: {self.IMPUTATION_TYPES}."
            )

    def correct_outlier(
        self,
        *,
        ecg: pd.DataFrame,
        rpeaks: pd.DataFrame,
        outlier_detection_results: pd.DataFrame | Sequence[pd.DataFrame],
    ):
        """Correct outliers in the R-peak data.

        Parameters
        ----------
        ecg : :class:`~pandas.DataFrame`
            The ECG data.
        rpeaks : :class:`~pandas.DataFrame`
            The R-peak data.
        outlier_detection_results : list of :class:`~pandas.DataFrame`
            The results of the outlier detection algorithms.

        """
        self._check_imputation_type_valid()
        rpeaks = rpeaks.copy()
        # get the last sample because it will get lost when computing the RR interval
        # last_sample = rpeaks.iloc[-1]

        if isinstance(outlier_detection_results, Sequence):
            outlier_detection_results = pd.concat(outlier_detection_results, axis=1)
        outlier_detection_results = outlier_detection_results.any(axis=1)

        # mark all outliers in the rpeaks dataframe
        removed_beats = rpeaks.loc[outlier_detection_results.squeeze()]

        rpeaks = rpeaks.assign(r_peak_outlier=0)
        rpeaks.loc[outlier_detection_results] = pd.NA
        rpeaks = rpeaks.fillna({"r_peak_outlier": 1.0})

        if ecg is not None:
            # also mark outlier in the ECG signal dataframe
            ecg = ecg.assign(r_peak_outlier=0)
            ecg.loc[removed_beats["r_peak_time"], "r_peak_outlier"] = 1.0

        # replace the last beat by average
        # if "R_Peak_Quality" in rpeaks.columns:
        #     rpeaks.loc[last_sample.name] = [
        #         rpeaks["R_Peak_Quality"].mean(),
        #         last_sample["R_Peak_Idx"],
        #         rpeaks["RR_Interval"].mean(),
        #         0.0,
        #     ]

        if self.imputation_type == "moving_average":
            rpeaks["rr_interval_ms"] = rpeaks["rr_interval_ms"].fillna(
                rpeaks["rr_interval_ms"]
                .rolling(self.imputation_params.get("window_size", 21), center=True, min_periods=0)
                .mean()
            )

        # interpolate all columns (except rr_interval_ms if imputation_type is moving average)
        rpeaks = rpeaks.interpolate(method="linear", limit_direction="both")
        # drop duplicate R peaks (can happen during outlier correction at edge cases)
        rpeaks = rpeaks.drop_duplicates(subset="r_peak_sample")

        rpeaks = rpeaks.assign(heart_rate_bpm=(60 / rpeaks["rr_interval_ms"]) * 1000)
        heart_rate_interpolated = nk.signal_interpolate(
            x_values=np.squeeze(rpeaks["r_peak_sample"].values),
            y_values=np.squeeze(rpeaks["heart_rate_bpm"].values),
            x_new=np.arange(0, len(ecg)),
        )
        ecg = ecg.assign(heart_rate_bpm=heart_rate_interpolated)

        rpeaks = rpeaks.assign(r_peak_sample=rpeaks["r_peak_sample"].round())
        # reindex
        rpeaks = rpeaks.reset_index(drop=True)
        # rpeaks = rpeaks.reset_index()
        # rpeaks = rpeaks.rename(columns={"heartbeat_id": "heartbeat_id_original"})
        rpeaks.index.name = "heartbeat_id"

        # reorder columns
        rpeaks = rpeaks[
            [
                "r_peak_time",
                "r_peak_sample",
                "rr_interval_ms",
                "heart_rate_bpm",
                "r_peak_quality",
                "r_peak_outlier",
                # "heartbeat_id_original",
            ]
        ]
        rpeaks = rpeaks.astype(
            {
                "r_peak_sample": "Int64",
                "rr_interval_ms": "Float64",
                "heart_rate_bpm": "Float64",
                "r_peak_quality": "Float64",
                "r_peak_outlier": "Int64",
                # "heartbeat_id_original": "Int64",
            }
        )

        self.ecg_processed_ = ecg
        self.points_ = rpeaks
