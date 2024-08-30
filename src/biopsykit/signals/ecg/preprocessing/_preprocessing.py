from typing import Optional

import neurokit2 as nk
import pandas as pd


def clean_ecg(raw_signal: pd.Series, sampling_rate_hz: int, method: Optional[str] = "biosppy") -> pd.Series:
    """Clean ECG signals using :func:`~neurokit2.ecg_clean`.

    Args:
        raw_signal: pd.Series containing the raw ECG signal
        sampling_rate_hz: sampling rate of the ECG signal in hz
        method: cleaning method (default is "biosppy"), can be any either "neurokit" or "biosppy"

    Returns
    -------
        clean_signal: pd.DataFrame containing filtered signal while keeping index of input signal
    """
    if method not in ["neurokit", "biosppy"]:
        raise ValueError("Not implemented yet!")

    clean_signal = nk.ecg_clean(raw_signal, sampling_rate=sampling_rate_hz, method=method)
    clean_signal = pd.Series(clean_signal, index=raw_signal.index, name="ecg")

    return clean_signal
