import neurokit2 as nk
import numpy as np
import pandas as pd
from tpcp import Algorithm

__all__ = ["RPeakOutlierCorrectionHrvLipponen2019"]


class RPeakOutlierCorrectionHrvLipponen2019(Algorithm):
    r"""R-peak outlier correction algorithm based on Lipponen and Tarvainen (2019) [Lip19]_.

    This algorithm identifies and corrects erroneous peak placements based on outliers in peak-to-peak differences.

    .. warning ::
        This algorithm might *add* additional R peaks or *remove* certain ones, so results of this function
        might **not** match with the previously detected R peaks anymore. Thus, R peaks resulting from this might not
        be used in combination with the raw ECG signal for further processing.


    Attributes
    ----------
    points_ : :class:`~pandas.DataFrame`
        The corrected R-peak data. The DataFrame contains the corrected R-peak locations, with the index
        representing the heartbeat IDs and the column "r_peak_sample" containing the corrected R-peak sample


    References
    ----------
    .. [Lip19] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for heart rate variability time series
        artefact correction using novel beat classification. Journal of Medical Engineering and Technology, 43(3),
        173-181. https://doi.org/10.1080/03091902.2019.1640306

    """

    _action_methods = "correct_outlier"

    points_: pd.DataFrame

    def __init__(self):
        """Initialize new ``RPeakOutlierCorrectionHrvLipponen2019`` algorithm instance."""
        super().__init__()

    def correct_outlier(
        self,
        *,
        rpeaks: pd.DataFrame,
        sampling_rate_hz: float,
    ):
        """Correct outliers in the R-peak data.

        Parameters
        ----------
        rpeaks : :class:`~pandas.DataFrame`
            The R-peak data. The DataFrame contains the R-peak locations, with the index
            representing the heartbeat IDs and the column "r_peak_sample" containing the R-peak samples.
        sampling_rate_hz : float
            The sampling rate of the ECG signal in Hz.

        """
        # fill missing RR intervals with interpolated R Peak Locations
        rpeaks_corrected = (rpeaks["rr_interval_ms"].cumsum() * sampling_rate_hz / 1000).astype(int)
        rpeaks_corrected = np.append(
            rpeaks["r_peak_sample"].iloc[0], rpeaks_corrected.iloc[:-1] + rpeaks["r_peak_sample"].iloc[0]
        )
        _, rpeaks_corrected = nk.signal_fixpeaks(
            rpeaks_corrected, int(sampling_rate_hz), iterative=False, method="Kubios"
        )
        rpeaks_corrected = rpeaks_corrected.astype(int)
        rpeaks_result = pd.DataFrame(rpeaks_corrected, columns=["r_peak_sample"])
        rpeaks_result.index.name = "heartbeat_id"

        self.points_ = rpeaks_result
        return self
