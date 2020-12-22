from typing import Union

import numpy as np
from scipy import signal
from scipy import interpolate


def imu_norm(data: np.ndarray) -> np.ndarray:
    return np.linalg.norm(data, axis=1)


def downsample(data: np.ndarray, sampling_rate: Union[int, float],
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
