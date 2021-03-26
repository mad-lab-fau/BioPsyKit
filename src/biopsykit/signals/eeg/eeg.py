from typing import Optional, Dict, Sequence

import pandas as pd
import numpy as np


def relative_band_energy(data: pd.DataFrame, sampling_rate: Optional[int] = 250,
                             freq_bands: Optional[Dict[str, Sequence[int]]] = None) -> pd.DataFrame:
    from mne.time_frequency import psd_array_welch

    raw_array = np.transpose(data.values)
    # define the different frequency bands
    if freq_bands is None:
        freq_bands = {
            # "delta": [1, 4],
            "theta": [4, 8],
            "alpha": [8, 13],
            "beta": [13, 30],
            "gamma": [30, 44]
        }

    # compute power spectral density using Welch's method
    psds, freqs = psd_array_welch(raw_array, sfreq=sampling_rate, fmin=0.5, fmax=sampling_rate/2, average=None,
                                  n_overlap=int(0.9 * sampling_rate), verbose=0)

    eeg_bands = []
    for fmin, fmax in freq_bands.values():
        # extract the FFT coefficients of the respective frequency band
        psds_band = psds[:, (freqs >= fmin) & (freqs < fmax)]
        # sum over the frequency bins
        psds_band = psds_band.sum(axis=1)
        # mean over the EEG channels
        psds_band = psds_band.mean(axis=0)
        eeg_bands.append(psds_band)

    eeg_bands = np.array(eeg_bands)

    # convert into dataframe
    df_bands = pd.DataFrame(np.transpose(eeg_bands), columns=list(freq_bands.keys()))
    # divide the band coefficients by the total sum of all frequency
    # band coefficients per sample to get the relative band powers
    df_bands = df_bands.div(df_bands.sum(axis=1), axis=0)

    # create a time axis and set as new index
    df_bands['timestamp'] = pd.to_datetime(
        np.linspace(float(data.index[0].to_numpy()), float(data.index[-1].to_numpy()), len(df_bands)))
    df_bands.set_index('timestamp', inplace=True)
    df_bands = df_bands.tz_localize("UTC").tz_convert("Europe/Berlin")
    return df_bands
