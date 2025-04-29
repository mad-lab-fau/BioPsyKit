import pandas as pd

from biopsykit.signals.ecg.outlier_correction._base_outlier_detection import BaseRPeakOutlierDetection

__all__ = ["RPeakOutlierDetectionPhysiological"]


class RPeakOutlierDetectionPhysiological(BaseRPeakOutlierDetection):
    """R-peak outlier detection algorithm through physiological limits.

    This algorithm detects outliers in R-peak data using physiological limits based on the RR intervals.

    """

    hr_thresholds: tuple[int, int]

    def __init__(self, hr_thresholds: tuple[int, int] = (45, 200)) -> None:
        """Initialize new ``RPeakOutlierDetectionPhysiological`` algorithm instance.

        Parameters
        ----------
        hr_thresholds : tuple of int, optional
            Tuple containing the lower and upper heart rate thresholds in beats per minute (bpm).
            Default: (45, 200)

        """
        self.hr_thresholds = hr_thresholds
        super().__init__()

    def _assert_params(self):
        if len(self.hr_thresholds) != 2:
            raise ValueError(f"hr_thresholds must be a tuple of two integers. Got {self.hr_thresholds}.")
        if self.hr_thresholds[0] >= self.hr_thresholds[1]:
            raise ValueError(
                f"The first element of hr_thresholds must be less than the second element. Got {self.hr_thresholds}."
            )
        if self.hr_thresholds[0] <= 0 or self.hr_thresholds[1] <= 0:
            raise ValueError(f"The elements of hr_thresholds must be positive integers. Got {self.hr_thresholds}.")

    def detect_outlier(self, *, ecg: pd.DataFrame, rpeaks: pd.DataFrame, sampling_rate_hz: float):  # noqa: ARG002
        self._assert_params()

        outlier = (rpeaks["rr_interval_ms"] / 1000 > (60 / self.hr_thresholds[0])) | (
            (rpeaks["rr_interval_ms"] / 1000) < (60 / self.hr_thresholds[1])
        )
        outlier = outlier.to_frame(name="is_outlier")

        self.points_ = outlier
