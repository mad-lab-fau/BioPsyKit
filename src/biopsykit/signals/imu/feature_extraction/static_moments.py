"""Extract features from static moments of IMU data."""
from typing import Optional, Union, Tuple

import pandas as pd
import numpy as np
from scipy.stats import skew

from biopsykit.utils.array_handling import sanitize_input_nd
from biopsykit.utils.time import tz


def compute_features(
    data: pd.DataFrame,
    static_moments: pd.DataFrame,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    index: Optional[Union[int, str]] = None,
    timezone: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Compute features based on frequency and duration of static moments in given input signal.

    This function computes the following features:

    * ``sm_number``: number of static moments in data
    * ``sm_max``: maximum duration of static moments, i.e., longest duration
    * ``sm_max_position``: location of the beginning of the longest static moment in the input data normalized to
      ``[0, 1]`` where 0 = ``start`` and 1 = ``end``
    * ``sm_median``: median duration of static moments
    * ``sm_mean``: mean duration of static moments
    * ``sm_std``: standard deviation of static moment durations
    * ``sm_skewness``: skewness of static moment durations

    The features are both computed on all detected static moments and on static moments that are longer than
    60 seconds (suffix ``_60``).


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        input data
    static_moments : :class:`~pandas.DataFrame`
        dataframe with beginning and end of static moments
    start : :class:`~pandas.Timestamp` or str, optional
        start timestamp in input data for feature extraction or ``None`` to set start index to the first index in
        ``data``. All samples *before* ``start`` will not be used for feature extraction.
    end : :class:`~pandas.Timestamp` or str, optional
        end timestamp in input data for feature extraction or ``None`` to set end index to the last index in
        ``data``. All samples *after* ``end`` will not be used for feature extraction.
    index : int or str, optional
        index label of the resulting dataframe or ``None`` to assign a default label (0)
    timezone : str, optional
        timezone of the recorded data or ``None`` to use default timezone ("Europe/Berlin")

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with extracted static moment features

    """
    if data.empty:
        return None

    start, end = _get_start_end(data, start, end, timezone)
    total_time = end - start

    static_moments = sanitize_input_nd(static_moments, ncols=2)

    durations = np.array([static_moment_duration(data, sequence) for sequence in static_moments])
    durations_60 = durations[durations >= 60]

    loc_max_moment = data.index[static_moments[np.argmax(durations)][0]]
    loc_max_moment_relative = (loc_max_moment - start) / total_time

    feature_dict = {"sm_max_position": loc_max_moment_relative}
    # feature_dict['sleep_bouts_number'.format(index)] = len(sleep_bouts)
    # feature_dict['wake_bouts_number'] = len(wake_bouts)

    # mean_orientations = mean_orientation(data, static_sequences)
    # dominant_orientation = mean_orientations.iloc[mean_orientations.index.argmax()]
    # dict_ori = {'sm_dominant_orientation_{}'.format(x): dominant_orientation.loc['acc_{}'.format(x)] for x
    #             in
    #             ['x', 'y', 'z']}
    # feature_dict.update(dict_ori)

    for dur, suffix in zip([durations, durations_60], ["", "_60"]):
        feature_dict["sm_number{}".format(suffix)] = len(dur)
        feature_dict["sm_max{}".format(suffix)] = np.max(dur)
        feature_dict["sm_median{}".format(suffix)] = np.median(dur)
        feature_dict["sm_mean{}".format(suffix)] = np.mean(dur)
        feature_dict["sm_std{}".format(suffix)] = np.std(dur, ddof=1)
        feature_dict["sm_skewness{}".format(suffix)] = skew(dur)

    if index is None:
        index = 0
    return pd.DataFrame(feature_dict, index=[index])


def _get_start_end(
    data: pd.DataFrame,
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    timezone: str,
) -> Tuple[Union[str, pd.Timestamp], Union[str, pd.Timestamp]]:
    if timezone is None:
        timezone = tz

    if start is None:
        start = data.index[0]
    if end is None:
        end = data.index[-1]

    start = _to_timestamp(start, timezone)
    end = _to_timestamp(end, timezone)
    return start, end


def _to_timestamp(date: Union[str, pd.Timestamp], timezone: str) -> pd.Timestamp:
    if isinstance(date, str):
        date = pd.Timestamp(date, tz=timezone)
    return date


def static_moment_duration(data: pd.DataFrame, start_end: np.array) -> float:
    """Compute duration of static moment.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        input data
    start_end : array
        start and end index of static moment to compute duration

    Returns
    -------
    float
        duration in seconds

    """
    return (data.index[start_end[1]] - data.index[start_end[0]]).total_seconds()


def mean_orientation(data: pd.DataFrame, static_moments: pd.DataFrame) -> pd.DataFrame:
    """Compute mean orientation of acceleration signal within static moment windows.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        input data
    static_moments : :class:`~pandas.DataFrame`
        dataframe with start and end indices of static moments

    Returns
    -------
    :class:`~pandas.DataFrame`
        mean orientation (x, y, z) of acceleration signal for each static moment window

    """
    static_moments = sanitize_input_nd(static_moments, 2)
    mean_orientations = [data.iloc[start_end[0] : start_end[1]] for start_end in static_moments]
    mean_orientations = {len(data): data.mean() for data in mean_orientations}
    mean_orientations = pd.DataFrame(mean_orientations).T
    # mean_orientations.rename(columns={'index': 'length'}, inplace=True)
    return mean_orientations
