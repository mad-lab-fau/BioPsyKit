import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import median_abs_deviation
from tpcp import Algorithm

__all__ = ["BaseBPointOutlierCorrection"]

from biopsykit.utils.dtypes import BPointDataFrame, CPointDataFrame


class BaseBPointOutlierCorrection(Algorithm):
    """Base class for outlier correction algorithms for B-Point data.

    This class provides a template for outlier correction algorithms for B-Point data. It includes methods for
    detecting outliers and stationarizing B-Point data. The actual outlier correction algorithm must be implemented in
    a subclass.

    Attributes
    ----------
    points_ : :class:`~pandas.DataFrame`
        DataFrame containing the B-Point data with the outliers corrected. Each row contains the B-point location (in
        samples from beginning of signal) for each heartbeat, index functions as id of heartbeat. B-point locations can
        be NaN if no B-points were detected for certain heartbeats.

    """

    _action_methods = "correct_outlier"

    points_: BPointDataFrame

    def __init__(self) -> None:
        """Initialize new Outlier Correction Algorithm."""
        super().__init__()

    def correct_outlier(
        self,
        *,
        b_points: BPointDataFrame,
        c_points: CPointDataFrame | None,
        sampling_rate_hz: float,
        **kwargs,
    ):
        raise NotImplementedError("Method 'correct_outlier' must be implemented in a subclass!")

    @staticmethod
    def detect_b_point_outlier(stationary_data: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers in stationary B-Point data.

        Outliers are detected based on the median absolute deviation of the stationary data. If the difference between
        the stationary data and the median is greater than 3 times the median absolute deviation, the data point is
        considered an outlier.

        Parameters
        ----------
        stationary_data : :class:`~pandas.DataFrame`
            DataFrame containing the stationary B-Point data

        Returns
        -------
        :class:`~pandas.DataFrame`
            DataFrame containing the detected outliers

        """
        median = np.nanmedian(stationary_data["statio_data"])
        median_abs_dev = median_abs_deviation(stationary_data["statio_data"], axis=0, nan_policy="propagate")
        outliers = pd.DataFrame(index=stationary_data.index, columns=["outliers"])
        outliers["outliers"] = False
        outliers.loc[(stationary_data["statio_data"] - median) > (3 * median_abs_dev), "outliers"] = True
        outliers.loc[stationary_data["statio_data"].isna(), "outliers"] = True

        return outliers[outliers["outliers"]]

    @staticmethod
    def stationarize_b_points(b_points: pd.DataFrame, c_points: pd.DataFrame, sampling_rate_hz: float) -> pd.DataFrame:
        """Stationarize B-Point data by removing the baseline.

        The B-points are stationarized by removing the baseline from the distance to the C-point. The baseline is
        estimated using a 4th order low-pass Butterworth filter with a cutoff frequency of 0.1 Hz.

        Parameters
        ----------
        b_points : :class:`~pandas.DataFrame`
            Dataframe containing the extracted B-Points per heartbeat, index functions as id of heartbeat
        c_points : :class:`~pandas.DataFrame`
            Dataframe containing the extracted C-Points per heartbeat, index functions as id of heartbeat
        sampling_rate_hz : float
            Sampling rate of ICG signal in Hz

        Returns
        -------
        :class:`~pandas.DataFrame`
            DataFrame containing the stationarized B-Point data

        """
        b_point_sample = b_points["b_point_sample"].astype(float)
        c_point_sample = c_points["c_point_sample"].astype(float)
        dist_to_c_point = ((c_point_sample - b_point_sample) / sampling_rate_hz).to_frame()
        dist_to_c_point.columns = ["dist_to_c_point_ms"]
        dist_to_c_point["b_point_sample"] = b_point_sample
        dist_to_c_point = dist_to_c_point.interpolate().ffill().bfill()

        sos = butter(4, Wn=0.1, btype="lowpass", output="sos", fs=1)

        # padlen for sosfiltfilt according to scipy documentation
        padlen = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
        if len(dist_to_c_point) <= padlen:
            # signal needs to be at least as long as padlen => reduce padlen for shorter signals
            padlen = len(dist_to_c_point) - 1
        baseline = sosfiltfilt(sos, dist_to_c_point["dist_to_c_point_ms"].values, padlen=padlen)

        # TODO Check if necessary after changing to sos? If yes, find out how it needs to be changed
        #  => changed by changing padlen

        statio_data = (dist_to_c_point["dist_to_c_point_ms"] - baseline).to_frame()
        statio_data.columns = ["statio_data"]
        statio_data["b_point_sample"] = dist_to_c_point["b_point_sample"]
        statio_data["baseline"] = baseline
        statio_data = statio_data.assign(padlen=padlen)

        return statio_data
