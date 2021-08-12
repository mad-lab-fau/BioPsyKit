"""Module for processing EEG data."""

from typing import Optional, Dict, Sequence, Union

import pandas as pd
import numpy as np

from tqdm.auto import tqdm

from biopsykit.signals._base import _BaseProcessor


class EegProcessor(_BaseProcessor):
    """Class for processing EEG data."""

    def __init__(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        sampling_rate: Optional[float] = None,
        time_intervals: Optional[Union[pd.Series, Dict[str, Sequence[str]]]] = None,
        include_start: Optional[bool] = False,
    ):
        """Initialize an ``EegProcessor`` instance.

        You can either pass a data dictionary 'data_dict' containing EEG data or dataframe containing
        EEG data. For the latter, you can additionally supply time information via ``time_intervals`` parameter
        to automatically split the data into single phases.


        Parameters
        ----------
        data : :class:`~pandas.DataFrame` or dict
            dataframe (or dict of such) with EEG data
        sampling_rate : float, optional
            sampling rate of recorded data
        time_intervals : dict or :class:`~pandas.Series`, optional
            time intervals indicating how ``data`` should be split.
            Can either be a :class:`~pandas.Series` with the `start` times of the single phases
            (the phase names are then derived from the index) or a dictionary with tuples indicating
            `start` and `end` times of phases (the phase names are then derived from the dict keys).
            Default: ``None`` (data is not split further)
        include_start : bool, optional
            ``True`` to include the data from the beginning of the recording to the first time interval as the
            first phase (then named ``Start``), ``False`` otherwise. Default: ``False``

        """
        super().__init__(
            data=data, sampling_rate=sampling_rate, time_intervals=time_intervals, include_start=include_start
        )

        self.eeg_result: Dict[str, pd.DataFrame] = {}
        """Dictionary with EEG processing result dataframes, split into different phases.

        """

    def relative_band_energy(
        self,
        freq_bands: Optional[Dict[str, Sequence[int]]] = None,
        title: Optional[str] = None,
    ) -> None:
        """Process EEG signal.

        Parameters
        ----------
        freq_bands : dict
            dictionary with frequency bounds of EEG frequency bands. By default (``None``) the following
            frequency band definition (in Hz) is used:

            * ``theta``: [4, 8]
            * ``alpha``: [8, 13]
            * ``beta``: [13, 30]
            * ``gamma``: [30, 44]

        title : str, optional
            title of ECG processing progress bar in Jupyter Notebooks or ``None`` to leave empty. Default: ``None``

        """
        from mne.time_frequency import psd_array_welch  # pylint:disable=import-outside-toplevel

        eeg_result = {}
        for key, df in tqdm(self.data.items(), desc=title):

            raw_array = np.transpose(df.values)
            # define the different frequency bands
            if freq_bands is None:
                freq_bands = {
                    # "delta": [1, 4],
                    "theta": [4, 8],
                    "alpha": [8, 13],
                    "beta": [13, 30],
                    "gamma": [30, 44],
                }

            # compute power spectral density using Welch's method
            psds, freqs = psd_array_welch(
                raw_array,
                sfreq=self.sampling_rate,
                fmin=0.5,
                fmax=self.sampling_rate / 2,
                average=None,
                n_overlap=int(0.9 * self.sampling_rate),
                verbose=0,
            )

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
            df_bands["timestamp"] = pd.to_datetime(
                np.linspace(
                    float(df.index[0].to_numpy()),
                    float(df.index[-1].to_numpy()),
                    len(df_bands),
                )
            )
            df_bands = df_bands.set_index("timestamp")
            df_bands = df_bands.tz_localize("UTC").tz_convert("Europe/Berlin")
            eeg_result[key] = df_bands

        self.eeg_result = eeg_result
