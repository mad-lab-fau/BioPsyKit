from typing import Optional

import pandas as pd
from scipy import signal


def clean_icg_deriv(
    raw_signal: pd.Series, sampling_rate_hz: int, filter_type: Optional[str] = "butterworth"
) -> pd.Series:
    """Clean ICG dZ/dt signal using a band-pass filter.

    This function filters the raw dZ/dt ICG signal using a band-pass filter. The following filters are available:
        * Butterworth band-pass filter: 4th order, low cutoff 0.5 Hz, high cutoff 25 Hz, see Forouzanfar 2019
        * Elliptic band-pass filter (Cauer filter): 2nd order, low cutoff 0.75 Hz, high cutoff 40 Hz, pass-band ripple
            1 dB, stop-band ripple 80 dB, see Nabian 2017 & Ostadabbas 2023
        * Savitzky-Golay filter: low-pass, 3rd order polynomial, see Salah 2020 & Salah 2017 (only useful for signal
            with high frequency noise?)

    Parameters
    ----------
    raw_signal: pd.Series
        pd.Series containing the raw dZ/dt ICG signal
    sampling_rate_hz: int
        sampling rate of ICG dZ/dt signal in hz
    filter_type: str, optional
        type of filter. Must be one of "butterworth", "elliptic", or "savgol". Default: "butterworth"

    Returns
    -------
    pd.Series
        cleaned ICG dZ/dt signal

    """
    if filter_type not in ["butterworth", "elliptic", "savgol"]:
        raise ValueError("Filter type can only be 'butterworth', 'elliptic', or 'savgol'")

    if filter_type == "butterworth":
        sos = signal.butter(N=4, Wn=[0.5, 25], btype="bandpass", output="sos", fs=sampling_rate_hz)
        clean_signal = signal.sosfiltfilt(sos, raw_signal)

    elif filter_type == "elliptic":
        rp = 1.0
        rs = 80.0
        sos = signal.ellip(
            N=2, rp=rp, rs=rs, Wn=[0.75, 40], btype="bandpass", output="sos", fs=sampling_rate_hz, analog=False
        )
        clean_signal = signal.sosfiltfilt(sos, raw_signal)

    elif filter_type == "savgol":  # Savitzky-Golay filter (for high frequency noise?!)
        clean_signal = signal.savgol_filter(raw_signal, window_length=61, polyorder=3)

    else:
        raise ValueError("Filter type can only be 'butterworth', 'elliptic', or 'savgol'")

    clean_signal = pd.Series(clean_signal, index=raw_signal.index, name="icg_der")
    return clean_signal
