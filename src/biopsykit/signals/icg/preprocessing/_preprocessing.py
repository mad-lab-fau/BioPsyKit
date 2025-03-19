import pandas as pd
from scipy import signal
from tpcp import Algorithm, Parameter

from biopsykit.utils.array_handling import sanitize_input_dataframe_1d
from biopsykit.utils.dtypes import is_icg_raw_dataframe


class IcgPreprocessingBandpass(Algorithm):
    """Preprocessing algorithm for ICG signals using a band-pass filter.

    This algorithm applies a band-pass filter to the ICG signal to clean it from noise.

    Parameters
    ----------
    method : str, optional
        Method to use for filtering. Can be one of {"butterworth", "elliptic", "savgol"}. Default: "butterworth"

    Attributes
    ----------
    icg_clean_ : :class:`~pandas.DataFrame`
        Cleaned ICG signal

    """

    _action_methods = "clean"

    method: Parameter[str]

    icg_clean_: pd.DataFrame

    def __init__(self, method: str = "butterworth"):
        """Initialize new ``IcgPreprocessingBandpass`` instance.

        Parameters
        ----------
        method : str, optional
            Method to use for filtering. Can be one of {"butterworth", "elliptic", "savgol"}. Default: "butterworth"

        """
        self.method = method

    def clean(self, *, icg: pd.DataFrame, sampling_rate_hz: int):
        """Clean ICG signal using a band-pass filter.

        Parameters
        ----------
        icg : :class:`~pandas.DataFrame`
            pandas DataFrame containing the raw ICG signal
        sampling_rate_hz : int
            Sampling rate of the ICG signal in Hz

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the filter type is not one of {"butterworth", "elliptic", "savgol"}

        """
        is_icg_raw_dataframe(icg)
        icg = sanitize_input_dataframe_1d(icg, column="icg_der")
        icg = icg.squeeze()

        if icg.empty:
            self.icg_clean_ = pd.DataFrame(index=icg.index, columns=["icg_der"])
            return self

        if self.method not in ["butterworth", "elliptic", "savgol"]:
            raise ValueError("Filter type can only be 'butterworth', 'elliptic', or 'savgol'")

        if self.method == "butterworth":
            sos = signal.butter(N=4, Wn=[0.5, 25], btype="bandpass", output="sos", fs=sampling_rate_hz)
            clean_signal = signal.sosfiltfilt(sos, icg)
        elif self.method == "elliptic":
            rp = 1.0
            rs = 80.0
            sos = signal.ellip(
                N=2, rp=rp, rs=rs, Wn=[0.75, 40], btype="bandpass", output="sos", fs=sampling_rate_hz, analog=False
            )
            clean_signal = signal.sosfiltfilt(sos, icg)
        elif self.method == "savgol":  # Savitzky-Golay filter (for high frequency noise?!)
            clean_signal = signal.savgol_filter(icg, window_length=61, polyorder=3)
        else:
            raise ValueError("This should not happen")

        self.icg_clean_ = pd.DataFrame(clean_signal, index=icg.index, columns=["icg_der"])
        return self
