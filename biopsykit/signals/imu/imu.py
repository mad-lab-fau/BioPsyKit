from typing import Union, Optional

import pandas as pd
import numpy as np

import biopsykit.signals.utils as su
from biopsykit.signals.static_moment_detection import find_static_sequences


def convert_acc_data_to_g(data: Union[pd.DataFrame], inplace: Optional[bool] = False) -> Union[None, pd.DataFrame]:
    acc_cols = data.filter(like='acc').columns
    if not inplace:
        data = data.copy()
    data.loc[:, acc_cols] = data.loc[:, acc_cols] / 9.81

    if not inplace:
        return data


def get_windows(data: Union[np.array, pd.Series, pd.DataFrame],
                window_samples: Optional[int] = None,
                window_sec: Optional[int] = None,
                sampling_rate: Optional[Union[int, float]] = 0,
                overlap_samples: Optional[int] = None,
                overlap_percent: Optional[float] = None) -> pd.DataFrame:
    index = None
    index_resample = None
    if isinstance(data, (pd.DataFrame, pd.Series)):
        index = data.index

    data_window = su.sliding_window(data,
                                    window_samples=window_samples, window_sec=window_sec,
                                    sampling_rate=sampling_rate, overlap_samples=overlap_samples,
                                    overlap_percent=overlap_percent)
    if index is not None:
        index_resample = su.sliding_window(index.values,
                                           window_samples=window_samples, window_sec=window_sec,
                                           sampling_rate=sampling_rate, overlap_samples=overlap_samples,
                                           overlap_percent=overlap_percent)[:, 0]
        if isinstance(index, pd.DatetimeIndex):
            index_resample = pd.DatetimeIndex(index_resample)
            index_resample = index_resample.tz_localize('UTC').tz_convert(index.tzinfo)

    data_window = np.transpose(data_window)
    data_window = {axis: pd.DataFrame(np.transpose(data), index=index_resample) for axis, data in
                   zip(['x', 'y', 'z'], data_window)}
    data_window = pd.concat(data_window, axis=1)
    data_window.columns.names = ['axis', 'samples']
    return data_window


def get_var_norm(data: pd.DataFrame) -> pd.DataFrame:
    var = data.groupby(axis=1, level='axis').apply(lambda x: np.var(x, axis=1))
    norm = pd.DataFrame(np.linalg.norm(var, axis=1), index=var.index, columns=['var_norm'])
    return norm


def get_static_sequences(
        data: pd.DataFrame,
        threshold: float,
        window_samples: Optional[int] = None,
        window_sec: Optional[int] = None,
        sampling_rate: Optional[Union[int, float]] = 0,
        overlap_samples: Optional[int] = None,
        overlap_percent: Optional[float] = None,
) -> pd.DataFrame:
    # compute the data_norm of the variance in the windows
    window, overlap = su.sanitize_sliding_window_input(
        window_samples=window_samples, window_sec=window_sec,
        sampling_rate=sampling_rate, overlap_samples=overlap_samples, overlap_percent=overlap_percent
    )
    start_end = find_static_sequences(data, window_length=window, overlap=overlap, inactive_signal_th=threshold,
                                      metric='variance')
    if start_end[-1, -1] >= len(data):
        # fix: handle edge case manually
        start_end[-1, -1] = len(data) - 1
    return pd.DataFrame(start_end, columns=['start', 'end'])


def split_sequences(data: pd.DataFrame, n_splits: int):
    idx_split = np.arange(0, n_splits + 1) * (len(data) // n_splits)
    split_boundaries = list(
        zip(
            idx_split[:-1],
            idx_split[1:]
        )
    )
    return split_boundaries
