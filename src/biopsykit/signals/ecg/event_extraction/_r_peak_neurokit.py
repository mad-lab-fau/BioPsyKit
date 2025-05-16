import warnings

import neurokit2 as nk
import numpy as np
import pandas as pd

from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS, CanHandleMissingEventsMixin
from biopsykit.signals.ecg.event_extraction import BaseEcgExtraction
from biopsykit.utils.array_handling import sanitize_input_series
from biopsykit.utils.dtypes import EcgRawDataFrame, is_ecg_raw_dataframe

__all__ = ["RPeakExtractionNeurokit"]

from biopsykit.utils.exceptions import EcgProcessingError


class RPeakExtractionNeurokit(BaseEcgExtraction, CanHandleMissingEventsMixin):
    """Algorithm to extract R-peaks based on the NeuroKit2 algorithm.

    The R-peak is estimated using the NeuroKit2 algorithm, which is based on the Pan-Tompkins algorithm.

    For more information on the NeuroKit2 library, see [Mak21]_.

    References
    ----------
    .. [Mak21] Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lesspinasse, F., Pham, H., Schölzel, C., & S.H. Chen
        (2021). NeuroKit2: A Python Toolbox for Neurophysiological Signal Processing. Behavior Research Methods.
        https://doi.org/10.3758/s13428-020-01516-y

    """

    method: str

    ecg_processed_: pd.DataFrame

    def __init__(self, handle_missing_events: HANDLE_MISSING_EVENTS = "warn", method: str = "neurokit"):
        """Initialize new ``RPeakExtractionNeurokit`` algorithm instance.

        Parameters
        ----------
        handle_missing_events : one of {"warn", "raise", "ignore"}, optional
            How to handle missing data in the input dataframes. Default: "warn"

        """
        self.method = method
        super().__init__(handle_missing_events=handle_missing_events)

    def extract(
        self,
        *,
        ecg: EcgRawDataFrame,
        sampling_rate_hz: float,
    ):
        self._check_valid_missing_handling()
        is_ecg_raw_dataframe(ecg)
        ecg = sanitize_input_series(ecg, name="ecg")
        ecg = ecg.squeeze()

        try:
            ecg_processed, rpeak_idx = self._process_ecg(ecg, sampling_rate_hz=sampling_rate_hz)
        except ValueError as e:
            error_msg = f"Error processing ECG data: {e}"
            if self.handle_missing_events == "warn":
                warnings.warn(
                    f"{error_msg}. If you want to ignore this warning, set handle_missing_events to 'ignore'.",
                    UserWarning,
                )
            elif self.handle_missing_events == "raise":
                raise EcgProcessingError(error_msg) from e

        rpeaks = self._extract_r_peaks(ecg_processed, rpeak_idx, sampling_rate_hz=sampling_rate_hz)

        rpeaks = rpeaks.assign(heart_rate_bpm=(60 / rpeaks["rr_interval_ms"]) * 1000)
        heart_rate_interpolated = nk.signal_interpolate(
            x_values=np.squeeze(rpeaks["r_peak_sample"].values),
            y_values=np.squeeze(rpeaks["heart_rate_bpm"].values),
            x_new=np.arange(0, len(ecg_processed)),
        )
        ecg_processed = ecg_processed.assign(heart_rate_bpm=heart_rate_interpolated)

        # is_ecg_result_dataframe(ecg_processed)
        # is_r_peak_dataframe(rpeaks)
        self.ecg_processed_ = ecg_processed
        self.points_ = rpeaks

    def _process_ecg(self, ecg: pd.DataFrame, sampling_rate_hz: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        # find peaks using the specified method
        # instant_peaks: array indicating where detected R peaks are in the raw ECG signal
        # rpeak_idx array containing the indices of detected R peaks
        instant_peaks, rpeak_idx = nk.ecg_peaks(ecg, sampling_rate=int(sampling_rate_hz), method=self.method)

        rpeak_idx = pd.DataFrame(rpeak_idx["ECG_R_Peaks"], columns=["r_peak_sample"])
        rpeak_idx.index.name = "heartbeat_id"
        instant_peaks = np.squeeze(instant_peaks.to_numpy())

        if len(rpeak_idx) <= 3:
            error_msg = "Too few R peaks detected. Please check your ECG signal."
            if self.handle_missing_events == "warn":
                warnings.warn(
                    f"{error_msg}. If you want to ignore this warning, set handle_missing_events to 'ignore'.",
                    UserWarning,
                )
            elif self.handle_missing_events == "raise":
                raise EcgProcessingError(error_msg)

        # compute quality indicator
        quality = nk.ecg_quality(ecg, rpeaks=rpeak_idx.squeeze().to_numpy(), sampling_rate=int(sampling_rate_hz))

        # construct new dataframe
        ecg_result = pd.DataFrame(
            {
                "ecg": ecg,
                "r_peak_indicator": instant_peaks,
                "ecg_quality": quality,
                "r_peak_outlier": np.zeros(len(ecg)),
            },
            index=ecg.index,
        )
        ecg_result = ecg_result.astype(
            {"ecg": "Float64", "ecg_quality": "Float64", "r_peak_indicator": "Int64", "r_peak_outlier": "Int64"}
        )
        return ecg_result, rpeak_idx

    @staticmethod
    def _extract_r_peaks(ecg_result: pd.DataFrame, rpeak_idx: pd.DataFrame, sampling_rate_hz: float) -> pd.DataFrame:
        # copy new dataframe consisting of R peaks indices (and their respective quality indicator)
        rpeaks = ecg_result.loc[ecg_result["r_peak_indicator"] == 1.0, ["ecg_quality"]]
        rpeaks = rpeaks.rename(columns={"ecg_quality": "r_peak_quality"})
        rpeaks = rpeaks.assign(r_peak_sample=rpeak_idx.squeeze().to_numpy())
        # compute RR interval
        rpeaks = rpeaks.assign(rr_interval_ms=(np.ediff1d(rpeaks["r_peak_sample"], to_end=0) / sampling_rate_hz) * 1000)
        # ensure equal length by filling the last value with the average RR interval
        rpeaks.loc[rpeaks.index[-1], "rr_interval_ms"] = rpeaks["rr_interval_ms"].mean()
        rpeaks = rpeaks[["r_peak_sample", "rr_interval_ms", "r_peak_quality"]]
        rpeaks.index.name = "r_peak_time"
        rpeaks = rpeaks.reset_index()
        rpeaks.index.name = "heartbeat_id"

        rpeaks = rpeaks.astype({"r_peak_sample": int, "rr_interval_ms": float, "r_peak_quality": float})
        return rpeaks
