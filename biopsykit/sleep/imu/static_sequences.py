from typing import Sequence, Tuple, Dict, Optional, Union

import pandas as pd
import numpy as np
from scipy.stats import skew

from biopsykit.utils import sanitize_input_nd


def compute_static_sequences_features(data: pd.DataFrame,
                                      static_sequences: pd.DataFrame,
                                      sleep_endpoints: Dict):
    return static_sequences_features(data, static_sequences=static_sequences, sleep_endpoints=sleep_endpoints)


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

    total_sleep_time = (wake_onset - sleep_onset)

    static_sequences = sanitize_input_nd(static_sequences, ncols=2)

    durations = np.array([get_sequence_duration(data, sequence) for sequence in static_sequences])
    durations_60 = durations[durations >= 60]

    loc_max_sequence = data.index[static_sequences[np.argmax(durations)][0]]
    loc_max_sequence_relative = (loc_max_sequence - sleep_onset) / total_sleep_time

    feature_dict = {}
    feature_dict['ss_max_position'] = loc_max_sequence_relative
    # feature_dict['sleep_bouts_number'.format(index)] = len(sleep_bouts)
    # feature_dict['wake_bouts_number'] = len(wake_bouts)

    # mean_orientations = get_mean_orientation_in_static_sequences(data, static_sequences)
    # dominant_orientation = mean_orientations.iloc[mean_orientations.index.argmax()]
    # dict_ori = {'ss_dominant_orientation_{}'.format(x): dominant_orientation.loc['acc_{}'.format(x)] for x
    #             in
    #             ['x', 'y', 'z']}
    # feature_dict.update(dict_ori)

    for dur, suffix in zip([durations, durations_60], ['', '_60']):
        feature_dict['ss_number{}'.format(suffix)] = len(dur)
        feature_dict['ss_max{}'.format(suffix)] = np.max(dur)
        feature_dict['ss_median{}'.format(suffix)] = np.median(dur)
        feature_dict['ss_mean{}'.format(suffix)] = np.mean(dur)
        feature_dict['ss_std{}'.format(suffix)] = np.std(dur, ddof=1)
        feature_dict['ss_skewness{}'.format(suffix)] = skew(dur)

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
