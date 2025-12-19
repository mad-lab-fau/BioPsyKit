import pandas as pd
from tpcp import Algorithm


class BaseRPeakOutlierDetection(Algorithm):
    """Base class for outlier detection algorithms for R-peak data.

    This class provides a template for outlier detection algorithms for R-peak data.

    Attributes
    ----------
    points_ : :class:`~pandas.DataFrame`
        DataFrame containing the R-peak data with the outliers detected.

    """

    _action_methods = "detect_outlier"

    points_: pd.DataFrame

    def __init__(self) -> None:
        """Initialize new Outlier Correction Algorithm."""
        super().__init__()

    def detect_outlier(
        self,
        *,
        ecg: pd.DataFrame,
        rpeaks: pd.DataFrame,
        sampling_rate_hz: float,
    ):
        raise NotImplementedError("Method 'detect_outlier' must be implemented in a subclass!")
