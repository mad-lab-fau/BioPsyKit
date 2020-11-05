from typing import Optional, Sequence, Tuple

import pandas as pd
import numpy as np


def max_increase(data: pd.DataFrame, feature_name: Optional[str] = "cortisol",
                 remove_s0: Optional[bool] = True, percent: Optional[bool] = False) -> pd.DataFrame:
    # computes (absolute or relative) maximum increase between first sample and all others.

    if remove_s0:
        # We have a S0 sample => drop it
        data = data.drop(0, level='sample')

    data = data[[feature_name]].unstack()

    if percent:
        max_inc = 100 * (data.iloc[:, 1:].max(axis=1) / data.iloc[:, 0])
    else:
        max_inc = data.iloc[:, 1:].max(axis=1) - data.iloc[:, 0]

    return pd.DataFrame(max_inc, columns=[
        "{}_max_inc_percent".format(feature_name) if percent else "{}_max_inc".format(feature_name)],
                        index=max_inc.index)


def auc(data: pd.DataFrame, feature_name: Optional[str] = "cortisol",
        remove_s0: Optional[bool] = True, saliva_times: Optional[Sequence[int]] = None) -> pd.DataFrame:
    if saliva_times is None:
        # check if dataframe has 'time' column
        if 'time' in data:
            saliva_times = list(data['time'].unique())
        else:
            raise ValueError("No saliva times specified!")
    saliva_times = np.array(saliva_times)

    if remove_s0:
        # We have a S0 sample => drop it
        data = data.drop(0, level='sample')
        saliva_times = saliva_times[1:]

    data = data[[feature_name]].unstack()

    idxs_post = np.where(saliva_times >= 0)[0]
    data_post = data.iloc[:, idxs_post]

    auc = {
        'auc_g': np.trapz(data, saliva_times),
        'auc_i': np.trapz(data.sub(data.iloc[:, 0], axis=0), saliva_times),
        'auc_i_post': np.trapz(data_post.sub(data_post.iloc[:, 0], axis=0), saliva_times[idxs_post]),
    }
    return pd.DataFrame(auc, index=data.index).add_prefix("cortisol_")


def standard_features(data: pd.DataFrame, feature_name: Optional[str] = "cortisol") -> pd.DataFrame:
    group_cols = ['subject']
    if 'day' in data.index.names:
        group_cols.append('day')

    data = data[[feature_name]].groupby(group_cols).agg(
        [np.argmax, np.mean, pd.DataFrame.std, pd.DataFrame.skew, pd.DataFrame.kurt])
    data.add_prefix("{}_".format(feature_name))
    return data


def slope(data: pd.DataFrame, sample_idx: Tuple[int, int], feature_name: Optional[str] = "cortisol",
          saliva_times: Optional[Sequence[int]] = None) -> pd.DataFrame:
    if saliva_times is None:
        # check if dataframe has 'time' column
        if 'time' in data:
            saliva_times = list(data['time'].unique())
        else:
            raise ValueError("No saliva times specified!")
    saliva_times = np.array(saliva_times)

    data = data[[feature_name]].unstack()

    return pd.DataFrame(np.diff(data.iloc[:, sample_idx]) / np.diff(saliva_times[sample_idx]), index=data.index,
                        columns=['{}_slope{}{}'.format(feature_name, *sample_idx)])
