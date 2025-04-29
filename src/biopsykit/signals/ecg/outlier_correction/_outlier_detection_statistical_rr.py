import numpy as np
import pandas as pd

from biopsykit.signals.ecg.outlier_correction._base_outlier_detection import BaseRPeakOutlierDetection

__all__ = ["RPeakOutlierDetectionRRIntervalStatistics"]


class RPeakOutlierDetectionRRIntervalStatistics(BaseRPeakOutlierDetection):
    """R-peak outlier detection algorithm based on RR interval statistics.

    This class detects R-peak outliers based on statistical outlier of the RR intervals. If a RR interval is
    outside certain multiple of the standard deviation, it is considered an outlier.


    """

    rr_statistics_threshold: float

    def __init__(self, rr_statistics_threshold: float = 2.576) -> None:
        """Initialize new ``RPeakOutlierDetectionRRIntervalStatistics`` algorithm instance.

        Parameters
        ----------
        rr_statistics_threshold : float, optional
            Threshold for the RR interval statistics above which a beat is considered an outlier.
            Default: 2.576 (99% confidence interval)

        """
        self.rr_statistics_threshold = rr_statistics_threshold
        super().__init__()

    def _assert_params(self):
        if self.rr_statistics_threshold < 0:
            raise ValueError(
                f"rr_statistics_threshold must be a float greater than 0. Got {self.rr_statistics_threshold}."
            )

    def detect_outlier(self, *, ecg: pd.DataFrame, rpeaks: pd.DataFrame, sampling_rate_hz: float):  # noqa: ARG002
        self._assert_params()

        # compute z-score of RR intervals
        rri = rpeaks["rr_interval_ms"]
        z_score = np.abs((rri - np.nanmean(rri)) / np.nanstd(rri, ddof=1))

        outlier = z_score > self.rr_statistics_threshold
        outlier = outlier.to_frame(name="is_outlier")

        self.points_ = outlier
