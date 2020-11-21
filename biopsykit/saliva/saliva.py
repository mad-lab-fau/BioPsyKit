import warnings
from typing import Optional, Sequence, Tuple, Union

import pandas as pd
import numpy as np


def wide_to_long(data: pd.DataFrame, feature_name: Optional[str] = 'cortisol') -> pd.DataFrame:
    idx_col = ['subject']
    if 'condition' in data.index.names:
        idx_col.append('condition')
    if 'day' in data.index.names:
        idx_col.append('day')
    return pd.wide_to_long(data.reset_index(), stubnames=feature_name, sep='_',
                           i=idx_col, j='feature', suffix=r"\w+")


def saliva_mean_se(data: pd.DataFrame, feature_name: Optional[Union[str, Sequence[str]]] = 'cortisol',
                   remove_s0: Optional[bool] = True) -> pd.DataFrame:
    """Computes mean and standard error per saliva sample"""

    if isinstance(feature_name, list):
        dict_result = {}
        for feature in feature_name:
            dict_result[feature] = saliva_mean_se(data[[feature]], feature_name=feature, remove_s0=remove_s0)
        return dict_result

    if remove_s0:
        data = data.drop(0, level="sample")

    if 'time' in data:
        data_grp = data.groupby(["sample", 'condition']).apply(lambda df_sample: pd.Series(
            {'mean': df_sample[feature_name].mean(), 'se': df_sample[feature_name].std() / np.sqrt(len(df_sample)),
             'time': int(df_sample['time'].unique())}))
        data_grp = data_grp.set_index('time', append=True)
    else:
        data_grp = data.groupby(["sample", 'condition']).apply(lambda df_sample: pd.Series(
            {'mean': df_sample[feature_name].mean(), 'se': df_sample[feature_name].std() / np.sqrt(len(df_sample))}))
    return data_grp


def max_increase(data: pd.DataFrame, feature_name: Optional[Union[str, Sequence[str]]] = "cortisol",
                 remove_s0: Optional[bool] = True, percent: Optional[bool] = False) -> pd.DataFrame:
    # computes (absolute or relative) maximum increase between first sample and all others.
    _check_data_format(data)

    if isinstance(feature_name, list):
        dict_result = {}
        for feature in feature_name:
            dict_result[feature] = max_increase(data[[feature]], feature_name=feature, remove_s0=remove_s0,
                                                percent=percent)
        return dict_result

    if remove_s0:
        # We have a S0 sample => drop it
        data = data.drop(0, level='sample')

    if feature_name not in data:
        raise ValueError("No `{}` columns in data!".format(feature_name))
    data = data[[feature_name]].unstack()

    max_inc = (data.iloc[:, 1:].max(axis=1) - data.iloc[:, 0])
    if percent:
        max_inc = 100.0 * max_inc / np.abs(data.iloc[:, 0])

    out = pd.DataFrame(max_inc, columns=[
        "{}_max_inc_percent".format(feature_name) if percent else "{}_max_inc".format(feature_name)],
                       index=max_inc.index)
    out.columns.name = "feature"
    return out


def auc(data: pd.DataFrame, feature_name: Optional[Union[str, Sequence[str]]] = "cortisol",
        remove_s0: Optional[bool] = True, saliva_times: Optional[Sequence[int]] = None) -> pd.DataFrame:
    # TODO IMPORTANT: saliva_time '0' is defined as "right before stress" (0 min of stress)
    # => auc_post means all saliva times after beginning of stress (>= 0)

    _check_data_format(data)
    saliva_times = _get_saliva_times(data, saliva_times, remove_s0)
    _check_saliva_times(saliva_times)

    if isinstance(feature_name, list):
        dict_result = {}
        for feature in feature_name:
            dict_result[feature] = auc(data[[feature]], feature_name=feature, remove_s0=remove_s0,
                                       saliva_times=saliva_times)
        return dict_result

    if remove_s0:
        # We have a S0 sample => drop it
        data = data.drop(0, level='sample')

    if feature_name not in data:
        raise ValueError("No `{}` columns in data!".format(feature_name))
    data = data[[feature_name]].unstack()

    idxs_post = None
    if saliva_times.ndim == 1:
        idxs_post = np.where(saliva_times > 0)[0]
    elif saliva_times.ndim == 2:
        warnings.warn("Not computing `auc_i_post` values because this is only implemented if `saliva_times` "
                      "are the same for all subjects.")

    auc_data = {
        'auc_g': np.trapz(data, saliva_times),
        'auc_i': np.trapz(data.sub(data.iloc[:, 0], axis=0), saliva_times)
    }

    if idxs_post is not None:
        data_post = data.iloc[:, idxs_post]
        auc_data['auc_i_post'] = np.trapz(data_post.sub(data_post.iloc[:, 0], axis=0), saliva_times[idxs_post])

    out = pd.DataFrame(auc_data, index=data.index).add_prefix("{}_".format(feature_name))
    out.columns.name = "feature"
    return out


def standard_features(data: pd.DataFrame,
                      feature_name: Optional[Union[str, Sequence[str]]] = "cortisol") -> pd.DataFrame:
    group_cols = ['subject']

    _check_data_format(data)

    if isinstance(feature_name, list):
        dict_result = {}
        for feature in feature_name:
            dict_result[feature] = standard_features(data[[feature]], feature_name=feature)
        return dict_result

    if 'condition' in data.index.names:
        group_cols.append('condition')
    # also group by days and/or condition if we have multiple days present in the index
    if 'day' in data.index.names:
        group_cols.append('day')

    if feature_name not in data:
        raise ValueError("No `{}` columns in data!".format(feature_name))

    out = data[[feature_name]].groupby(group_cols).agg(
        [np.argmax, pd.DataFrame.mean, pd.DataFrame.std, pd.DataFrame.skew, pd.DataFrame.kurt])
    # drop 'feature_name' multiindex column and add as prefix to columns
    out.columns = out.columns.droplevel(0)
    out = out.add_prefix("{}_".format(feature_name))
    out.columns.name = "feature"
    return out


def slope(data: pd.DataFrame, sample_idx: Union[Tuple[int, int], Sequence[int]],
          feature_name: Optional[Union[str, Sequence[str]]] = "cortisol",
          saliva_times: Optional[Sequence[int]] = None) -> pd.DataFrame:
    _check_data_format(data)
    saliva_times = _get_saliva_times(data, saliva_times, remove_s0=False)
    _check_saliva_times(saliva_times)

    if feature_name not in data:
        raise ValueError("No `{}` columns in data!".format(feature_name))

    # ensure list
    sample_idx = list(sample_idx)

    if len(sample_idx) != 2:
        raise ValueError("Exactly 2 indices needed for computing slope. Got {} indices.".format(len(sample_idx)))

    if isinstance(feature_name, list):
        dict_result = {}
        for feature in feature_name:
            dict_result[feature] = slope(data[[feature]], sample_idx=sample_idx, feature_name=feature_name,
                                         saliva_times=saliva_times)

    data = data[[feature_name]].unstack()

    # replace idx values like '-1' with the actual index
    if sample_idx[0] < 0:
        sample_idx[0] = len(data.columns) + sample_idx[0]

    if sample_idx[1] < 0:
        sample_idx[1] = len(data.columns) + sample_idx[1]

    # check that second index is bigger than first index
    if sample_idx[0] >= sample_idx[1]:
        raise ValueError("`sample_idx[1]` must be bigger than `sample_idx[0]`. Got {}".format(sample_idx))

    if sample_idx[1] > (len(data.columns) - 1):
        raise ValueError("`sample_idx[1]` is out of bounds!")

    out = pd.DataFrame(np.diff(data.iloc[:, sample_idx]) / np.diff(saliva_times[sample_idx]), index=data.index,
                       columns=['{}_slope{}{}'.format(feature_name, *sample_idx)])
    out.columns.name = "feature"
    return out


def _check_data_format(data: pd.DataFrame):
    if data is None:
        raise ValueError("`data` must not be None!")
    if 'sample' not in data.index.names or data.index.nlevels <= 1:
        raise ValueError("`data` is expected in long-format with subject IDs ('subject', 0-n) as 1st level and "
                         "sample IDs ('sample', 0-m) as 2nd level!")


def _check_saliva_times(saliva_times: np.array):
    if np.any(np.diff(saliva_times) <= 0):
        raise ValueError("`saliva_times` must be increasing!")


def _get_saliva_times(data: pd.DataFrame, saliva_times: np.array, remove_s0: bool) -> np.array:
    if saliva_times is None:
        # check if dataframe has 'time' column
        if 'time' in data.columns:
            saliva_times = np.array(data.unstack()['time'])
            if np.all((saliva_times == saliva_times[0])):
                # all subjects have the same saliva times
                saliva_times = saliva_times[0]
        else:
            raise ValueError("No saliva times specified!")

    # ensure numpy
    saliva_times = np.array(saliva_times)

    if remove_s0:
        # check whether we have the same saliva times for all subjects (1d array) or not (2d array)
        if saliva_times.ndim == 1:
            saliva_times = saliva_times[1:]
        elif saliva_times.ndim == 2:
            saliva_times = saliva_times[:, 1:]
        else:
            raise ValueError("`saliva_times` has invalid dimensions: {}".format(saliva_times.ndim))

    return saliva_times
