from typing import Sequence, Tuple, Dict, Optional, Union

import pandas as pd
import numpy as np
from scipy.stats import skew

from biopsykit.utils import sanitize_input_nd


def compute_static_sequences_features(data: pd.DataFrame,
                                      static_sequences: pd.DataFrame,
                                      sleep_endpoints: Dict,
                                      split_boundaries: Optional[Sequence[Tuple[int, int]]] = None,
                                      ):
    sleep_bouts = sleep_endpoints['sleep_bouts']
    wake_bouts = sleep_endpoints['wake_bouts']

    if split_boundaries is None:
        return static_sequences_features(data, static_sequences=static_sequences, sleep_endpoints=sleep_endpoints)
    else:
        feature_list = [
            static_sequences_features(data, static_sequences=static_sequences, sleep_endpoints=sleep_endpoints,
                                      index="total")]
    static_sequences = sanitize_input_nd(static_sequences, ncols=2)

    for i, (start, end) in enumerate(split_boundaries):
        start_date = data.index[start].round('s')
        end_date = data.index[end].round('s')
        sb = [bout for bout in sleep_bouts['start'] if start_date <= bout <= end_date]
        wb = [bout for bout in wake_bouts['start'] if start_date <= bout <= end_date]
        intervals = static_sequences[np.logical_and(static_sequences[:, 0] >= start, static_sequences[:, 0] <= end)]
        feature_list.append(
            static_sequences_features(data, static_sequences=intervals, sleep_endpoints=sleep_endpoints,
                                      sleep_bouts=sb,
                                      wake_bouts=wb, index=i + 1))
    return pd.concat(feature_list)


def static_sequences_features(
        data: pd.DataFrame,
        static_sequences: pd.DataFrame,
        sleep_endpoints: Optional[Dict] = None,
        index: Optional[Union[int, str]] = None,
        **kwargs):
    if sleep_endpoints is None:
        sleep_endpoints = {}

    sleep_bouts = kwargs.get('sleep_bouts', sleep_endpoints.get('sleep_bouts', None))
    wake_bouts = kwargs.get('wake_bouts', sleep_endpoints.get('wake_bouts', None))
    sleep_onset = kwargs.get('sleep_onset', sleep_endpoints.get('sleep_onset', None))
    wake_onset = kwargs.get('wake_onset', sleep_endpoints.get('wake_onset', None))
    sleep_onset = pd.Timestamp(sleep_onset, tz="Europe/Berlin")
    wake_onset = pd.Timestamp(wake_onset, tz="Europe/Berlin")

    static_sequences = sanitize_input_nd(static_sequences, ncols=2)
    feature_dict = {}

    durations = np.array([get_sequence_duration(data, sequence) for sequence in static_sequences])
    durations_60 = durations[durations >= 60]

    loc_max_sequence = data.index[static_sequences[np.argmax(durations)][0]]
    loc_max_sequence_relative = (loc_max_sequence - sleep_onset) / (wake_onset - sleep_onset)

    mean_orientations = get_mean_orientation_in_static_sequences(data, static_sequences)
    dominant_orientation = mean_orientations.iloc[mean_orientations.index.argmax()]

    feature_dict['sleep_bouts_number'.format(index)] = len(sleep_bouts)
    feature_dict['wake_bouts_number'] = len(wake_bouts)
    feature_dict['static_sequence_max_position'] = loc_max_sequence_relative
    dict_ori = {'static_sequence_dominant_orientation_{}'.format(x): dominant_orientation.loc['acc_{}'.format(x)] for x
                in
                ['x', 'y', 'z']}
    feature_dict.update(dict_ori)

    for dur, suffix in zip([durations, durations_60], ['', '_60']):
        feature_dict['static_sequence_number{}'.format(suffix)] = len(dur)
        feature_dict['static_sequence_max{}'.format(suffix)] = np.max(dur)
        feature_dict['static_sequence_median{}'.format(suffix)] = np.median(dur)
        feature_dict['static_sequence_mean{}'.format(suffix)] = np.mean(dur)
        feature_dict['static_sequence_std{}'.format(suffix)] = np.std(dur, ddof=1)
        feature_dict['static_sequence_skewness{}'.format(suffix)] = skew(dur)

    if index is None:
        index = [0]
    features = pd.DataFrame(feature_dict, index=[index])

    return features


def get_sequence_duration(data: pd.DataFrame, start_end: np.array) -> float:
    return (data.index[start_end[1]] - data.index[start_end[0]]).total_seconds()


def get_mean_orientation_in_static_sequences(data: pd.DataFrame, static_sequences: pd.DataFrame) -> np.array:
    static_sequences = sanitize_input_nd(static_sequences, 2)
    mean_orientations = [data.iloc[start_end[0]:start_end[1]] for start_end in static_sequences]
    mean_orientations = {len(data): data.mean() for data in mean_orientations}
    mean_orientations = pd.DataFrame(mean_orientations).T
    # mean_orientations.rename(columns={'index': 'length'}, inplace=True)
    return mean_orientations
