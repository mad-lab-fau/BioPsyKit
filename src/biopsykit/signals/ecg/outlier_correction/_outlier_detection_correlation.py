import neurokit2 as nk
import pandas as pd

from biopsykit.signals.ecg.outlier_correction._base_outlier_detection import BaseRPeakOutlierDetection

__all__ = ["RPeakOutlierDetectionCorrelation"]


class RPeakOutlierDetectionCorrelation(BaseRPeakOutlierDetection):
    """R-peak outlier detection algorithm through cross-correlation between beats.

    This class detects outliers in R-peak data using cross-correlation between heartbeats. It segments the ECG signal
    into individual heartbeats, computes the average heartbeat, and then calculates the correlation coefficient between
    each heartbeat and the average. If the correlation coefficient is below a specified threshold, the heartbeat is
    considered an outlier.

    """

    correlation_threshold: float

    def __init__(self, correlation_threshold: float = 0.3) -> None:
        """Initialize new ``RPeakOutlierDetectionCorrelation`` algorithm instance.

        Parameters
        ----------
        correlation_threshold : float, optional
            Threshold for the cross-correlation coefficient below which a beat is considered an outlier.
            Default: 0.3

        """
        self.correlation_threshold = correlation_threshold
        super().__init__()

    def _assert_params(self):
        if not 0 <= self.correlation_threshold <= 1:
            raise ValueError(
                f"correlation_threshold must be a float between 0 and 1. Got {self.correlation_threshold}."
            )

    def detect_outlier(self, *, ecg: pd.DataFrame, rpeaks: pd.DataFrame, sampling_rate_hz: float):
        self._assert_params()

        # signal outlier
        # segment individual heart beats
        rpeaks_arr = rpeaks["r_peak_sample"].to_numpy()
        heartbeats = nk.ecg_segment(ecg["ecg"], rpeaks_arr, int(sampling_rate_hz))
        heartbeats = nk.epochs_to_df(heartbeats)
        heartbeats_pivoted = heartbeats.pivot(index="Time", columns="Label", values="Signal")
        heartbeats = heartbeats.set_index("Index")
        heartbeats = heartbeats.loc[heartbeats.index.intersection(rpeaks["r_peak_sample"])].sort_values(by="Label")
        heartbeats = heartbeats[~heartbeats.index.duplicated()]
        heartbeats_pivoted.columns = heartbeats.index

        # compute the average over all heart beats and compute the correlation coefficient between all beats and
        # the average
        mean_beat = heartbeats_pivoted.mean(axis=1)
        heartbeats_pivoted["mean"] = mean_beat
        corr_coeff = heartbeats_pivoted.corr()["mean"].abs().sort_values(ascending=True)
        corr_coeff = corr_coeff.drop(index="mean")
        corr_coeff.index.name = "r_peak_sample"

        outlier_beats = corr_coeff[corr_coeff < self.correlation_threshold]
        outlier = rpeaks["r_peak_sample"].isin(outlier_beats.index)
        outlier = outlier.to_frame(name="is_outlier")
        self.points_ = outlier
