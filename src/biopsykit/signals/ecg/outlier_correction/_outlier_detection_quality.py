import pandas as pd

from biopsykit.signals.ecg.outlier_correction._base_outlier_detection import BaseRPeakOutlierDetection

__all__ = ["RPeakOutlierDetectionQuality"]


class RPeakOutlierDetectionQuality(BaseRPeakOutlierDetection):
    """R-peak outlier detection algorithm based on signal quality.

    This class detects R-peak outliers based on the quality of the ECG signal.

    """

    quality_threshold: float

    def __init__(self, quality_threshold: float = 0.4) -> None:
        """Initialize new ``RPeakOutlierDetectionQuality`` algorithm instance.

        Parameters
        ----------
        quality_threshold : float, optional
            Threshold for the signal quality below which a beat is considered an outlier.
            Default: 0.4

        """
        self.quality_threshold = quality_threshold
        super().__init__()

    def _assert_params(self):
        if not 0 <= self.quality_threshold <= 1:
            raise ValueError(f"quality_threshold must be a float between 0 and 1. Got {self.quality_threshold}.")

    def detect_outlier(self, *, ecg: pd.DataFrame, rpeaks: pd.DataFrame, sampling_rate_hz: float):  # noqa: ARG002
        self._assert_params()

        # signal outlier
        # segment individual heart beats
        outlier = rpeaks["r_peak_quality"] < self.quality_threshold
        outlier = outlier.to_frame(name="is_outlier")
        self.points_ = outlier
