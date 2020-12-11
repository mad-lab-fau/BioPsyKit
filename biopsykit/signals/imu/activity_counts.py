from typing import Union

import numpy as np
from scipy import signal
from scipy import interpolate


class ActivityCounts:
    """Generate Activity Counts from raw IMU signals.

    ActiGraph Activity Counts are a unit used in many human activity studies.
    However, it can only be outputed by the official ActiGraph Software.
    The following implementation uses a reverse engineered version of the ActiGraph filter based on
    Brond JC et al. 2017 [1].


    [1] https://www.ncbi.nlm.nih.gov/pubmed/28604558
    """

    data = None
    sampling_rate = None

    activity_counts_ = None

    @staticmethod
    def _aliasing_filter(data: np.ndarray, sampling_rate: Union[int, float]) -> np.ndarray:
        sos = signal.butter(5, [0.01, 7], 'bp', fs=sampling_rate, output='sos')
        return signal.sosfiltfilt(sos, data)

    @staticmethod
    def _actigraph_filter(data: np.ndarray) -> np.ndarray:
        b = [0.04910898, -0.12284184, 0.14355788, -0.11269399, 0.05380374, -0.02023027, 0.00637785, 0.01851254,
             -0.03815411, 0.04872652, -0.05257721, 0.04784714, -0.04601483, 0.03628334, -0.01297681, -0.00462621,
             0.01283540, -0.00937622, 0.00344850, -0.00080972, -0.00019623]
        a = [1.00000000, -4.16372603, 7.57115309, -7.98046903, 5.38501191, -2.46356271, 0.89238142, 0.06360999,
             -1.34810513, 2.47338133, -2.92571736, 2.92983230, -2.78159063, 2.47767354, -1.68473849, 0.46482863,
             0.46565289, -0.67311897, 0.41620323, -0.13832322, 0.01985172]
        return signal.filtfilt(b, a, data)

    @staticmethod
    def _downsample(data: np.ndarray, sampling_rate: Union[int, float],
                    final_sampling_rate: Union[int, float]) -> np.ndarray:
        if (sampling_rate / final_sampling_rate) % 1 == 0:
            return signal.decimate(data, int(sampling_rate / final_sampling_rate))
        else:
            # aliasing filter
            b, a = signal.cheby1(N=8, rp=0.05, Wn=0.8 / (sampling_rate / final_sampling_rate))
            data_lp = signal.filtfilt(a=a, b=b, x=data)
            # interpolation
            x_old = np.linspace(0, len(data_lp), num=len(data_lp), endpoint=False)
            x_new = np.linspace(0, len(data_lp), num=int(len(data_lp) / (sampling_rate / final_sampling_rate)),
                                endpoint=False)
            interpol = interpolate.interp1d(x=x_old, y=data_lp)
            return interpol(x_new)

    @staticmethod
    def _truncate(data: np.ndarray) -> np.ndarray:
        upper_threshold = 2.13  # g
        lower_threshold = 0.068  # g
        data[data > upper_threshold] = upper_threshold
        data[data < lower_threshold] = 0
        return data

    @staticmethod
    def _digitize_8bit(data: np.ndarray) -> np.ndarray:
        max_val = 2.13  # g
        data //= max_val / (2 ** 7)
        return data

    @staticmethod
    def _accumulate_minute_bins(data: np.ndarray) -> np.ndarray:
        n_samples = 10 * 60
        #  Pad data at end to "fill" last bin
        padded_data = np.pad(data, (0, n_samples - len(data) % n_samples), 'constant', constant_values=0)
        return padded_data.reshape((len(padded_data) // n_samples, -1)).mean(axis=1)

    def calculate(self, data: np.ndarray, sampling_rate: Union[int, float]) -> 'ActivityCounts':
        self.data = data
        self.sampling_rate = sampling_rate
        tmp = self.data.copy()

        tmp = self._downsample(tmp, sampling_rate, 30)
        tmp = self._aliasing_filter(tmp, 30)
        tmp = self._actigraph_filter(tmp)
        tmp = self._downsample(tmp, 30, 10)
        tmp = np.abs(tmp)
        tmp = self._truncate(tmp)
        tmp = self._digitize_8bit(tmp)
        self.activity_counts_ = self._accumulate_minute_bins(tmp)
        return self
