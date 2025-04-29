import numpy as np
import pandas as pd

from biopsykit.signals.ecg.outlier_correction._base_outlier_detection import BaseRPeakOutlierDetection

__all__ = ["RPeakOutlierDetectionRRDiffIntervalStatistics"]


class RPeakOutlierDetectionRRDiffIntervalStatistics(BaseRPeakOutlierDetection):
    """R-peak outlier detection algorithm based on statistics of successive RR intervals.

    This class detects R-peak outliers based on statistical outlier of successive RR interval differences. If the
    z-score of the successive RR interval differences is outside certain multiple of the standard deviation, it is
    considered an outlier.

    """

    rr_diff_statistics_threshold: float

    def __init__(self, rr_diff_statistics_threshold: float = 1.96) -> None:
        """Initialize new ``RPeakOutlierDetectionRRDiffIntervalStatistics`` algorithm instance.

        Parameters
        ----------
        rr_diff_statistics_threshold : float, optional
            Threshold for the successive RR interval statistics above which a beat is considered an outlier.
            Default: 1.96 (95% confidence interval)

        """
        self.rr_diff_statistics_threshold = rr_diff_statistics_threshold
        super().__init__()

    def _assert_params(self):
        if self.rr_diff_statistics_threshold < 0:
            raise ValueError(
                f"rr_diff_statistics_threshold must be a float greater than 0. Got {self.rr_diff_statistics_threshold}."
            )

    def detect_outlier(self, *, ecg: pd.DataFrame, rpeaks: pd.DataFrame, sampling_rate_hz: float):  # noqa: ARG002
        self._assert_params()

        # compute z-score of RR intervals
        rri_diff = np.ediff1d(rpeaks["rr_interval_ms"], to_end=0)
        rri_diff = pd.Series(rri_diff, name="rri_diff_ms")
        z_score = np.abs((rri_diff - np.nanmean(rri_diff)) / np.nanstd(rri_diff, ddof=1))

        outlier = z_score > self.rr_diff_statistics_threshold
        outlier = outlier.to_frame(name="is_outlier")

        self.points_ = outlier
