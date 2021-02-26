import warnings
from typing import Optional, Sequence, Tuple, Union, Dict

import pandas as pd
import numpy as np


def wide_to_long(data: pd.DataFrame, biomarker_name: str, levels: Union[str, Sequence[str]],
                 sep: Optional[str] = '_') -> pd.DataFrame:
    if isinstance(levels, str):
        levels = [levels]

    data = data.filter(like=biomarker_name)
    # reverse level order because nested multi-level index will be constructed from back to front
    levels = levels[::-1]
    # iteratively build up long-format dataframe
    for i, level in enumerate(levels):
        stubnames = list(data.columns)
        # stubnames are everything except the last part separated by underscore
        stubnames = sorted(set(['_'.join(s.split('_')[:-1]) for s in stubnames]))
        data = pd.wide_to_long(data.reset_index(), stubnames=stubnames, i=['subject'] + levels[0:i], j=level,
                               sep=sep, suffix=r'\w+')

    # reorder levels and sort
    return data.reorder_levels(['subject'] + levels[::-1]).sort_index()


def saliva_mean_se(data: pd.DataFrame, biomarker_type: Optional[Union[str, Sequence[str]]] = 'cortisol',
                   remove_s0: Optional[bool] = True) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Computes mean and standard error per saliva sample"""

    if isinstance(biomarker_type, list):
        dict_result = {}
        for biomarker in biomarker_type:
            biomarker_cols = [biomarker]
            if 'time' in data:
                biomarker_cols = ['time'] + biomarker_cols
            dict_result[biomarker] = saliva_mean_se(data[biomarker_cols], biomarker_type=biomarker, remove_s0=remove_s0)
        return dict_result

    if remove_s0:
        data = data.drop(0, level='sample', errors='ignore')
        data = data.drop('0', level='sample', errors='ignore')
        data = data.drop('S0', level='sample', errors='ignore')

    group_cols = list(data.index.names)
    group_cols.remove('subject')

    if 'time' in data:
        data_grp = data.groupby(group_cols).apply(lambda df_sample: pd.Series(
            {'mean': df_sample[biomarker_type].mean(), 'se': df_sample[biomarker_type].std() / np.sqrt(len(df_sample)),
             'time': int(df_sample['time'].unique())}))
        data_grp = data_grp.set_index('time', append=True)
    else:
        data_grp = data.groupby(group_cols).apply(lambda df_sample: pd.Series(
            {'mean': df_sample[biomarker_type].mean(),
             'se': df_sample[biomarker_type].std() / np.sqrt(len(df_sample))}))
    return data_grp


def max_increase(data: pd.DataFrame, biomarker_type: Optional[Union[str, Sequence[str]]] = "cortisol",
                 remove_s0: Optional[bool] = True,
                 percent: Optional[bool] = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    # computes (absolute or relative) maximum increase between first sample and all others.
    _check_data_format(data)

    if isinstance(biomarker_type, list):
        dict_result = {}
        for biomarker in biomarker_type:
            biomarker_cols = [biomarker]
            if 'time' in data:
                biomarker_cols = ['time'] + biomarker_cols
            dict_result[biomarker] = max_increase(data[biomarker_cols], biomarker_type=biomarker, remove_s0=remove_s0,
                                                  percent=percent)
        return dict_result

    if remove_s0:
        # We have a S0 sample => drop it
        data = data.drop(0, level='sample', errors='ignore')
        data = data.drop('0', level='sample', errors='ignore')
        data = data.drop('S0', level='sample', errors='ignore')

    if biomarker_type not in data:
        raise ValueError("No `{}` columns in data!".format(biomarker_type))
    data = data[[biomarker_type]].unstack()

    max_inc = (data.iloc[:, 1:].max(axis=1) - data.iloc[:, 0])
    if percent:
        max_inc = 100.0 * max_inc / np.abs(data.iloc[:, 0])

    out = pd.DataFrame(max_inc, columns=[
        "{}_max_inc_percent".format(biomarker_type) if percent else "{}_max_inc".format(biomarker_type)],
                       index=max_inc.index)
    out.columns.name = "biomarker"
    return out


def auc(data: pd.DataFrame, biomarker_type: Optional[Union[str, Sequence[str]]] = "cortisol",
        remove_s0: Optional[bool] = True,
        compute_auc_post: Optional[bool] = False,
        saliva_times: Optional[Sequence[int]] = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    # TODO add documentation; IMPORTANT: saliva_time '0' is defined as "right before stress" (0 min of stress)
    # => auc_post means all saliva times after beginning of stress (>= 0)

    _check_data_format(data)
    saliva_times = _get_saliva_times(data, saliva_times, remove_s0)
    _check_saliva_times(saliva_times)

    if isinstance(biomarker_type, list):
        dict_result = {}
        for biomarker in biomarker_type:
            biomarker_cols = [biomarker]
            if 'time' in data:
                biomarker_cols = ['time'] + biomarker_cols
            dict_result[biomarker] = auc(data[biomarker_cols], biomarker_type=biomarker, remove_s0=remove_s0,
                                         saliva_times=saliva_times)
        return dict_result

    if remove_s0:
        # We have a S0 sample => drop it
        data = data.drop(0, level='sample', errors='ignore')
        data = data.drop('0', level='sample', errors='ignore')
        data = data.drop('S0', level='sample', errors='ignore')

    if biomarker_type not in data:
        raise ValueError("No `{}` columns in data!".format(biomarker_type))
    data = data[[biomarker_type]].unstack()

    auc_data = {
        'auc_g': np.trapz(data, saliva_times),
        'auc_i': np.trapz(data.sub(data.iloc[:, 0], axis=0), saliva_times)
    }

    if compute_auc_post:
        idxs_post = None
        if saliva_times.ndim == 1:
            idxs_post = np.where(saliva_times > 0)[0]
        elif saliva_times.ndim == 2:
            warnings.warn("Not computing `auc_i_post` values because this is only implemented if `saliva_times` "
                          "are the same for all subjects.")
        if idxs_post is not None:
            data_post = data.iloc[:, idxs_post]
            auc_data['auc_i_post'] = np.trapz(data_post.sub(data_post.iloc[:, 0], axis=0), saliva_times[idxs_post])

    out = pd.DataFrame(auc_data, index=data.index).add_prefix("{}_".format(biomarker_type))
    out.columns.name = "biomarker"
    return out


def standard_features(data: pd.DataFrame,
                      biomarker_type: Optional[Union[str, Sequence[str]]] = "cortisol") -> Union[
    pd.DataFrame, Dict[str, pd.DataFrame]]:
    group_cols = ['subject']

    _check_data_format(data)

    if isinstance(biomarker_type, list):
        dict_result = {}
        for biomarker in biomarker_type:
            biomarker_cols = [biomarker]
            if 'time' in data:
                biomarker_cols = ['time'] + biomarker_cols
            dict_result[biomarker] = standard_features(data[biomarker_cols], biomarker_type=biomarker)
        return dict_result

    if 'condition' in data.index.names:
        group_cols.append('condition')
    # also group by days and/or condition if we have multiple days present in the index
    if 'day' in data.index.names:
        group_cols.append('day')

    if biomarker_type not in data:
        raise ValueError("No `{}` columns in data!".format(biomarker_type))

    out = data[[biomarker_type]].groupby(group_cols).agg(
        [np.argmax, pd.DataFrame.mean, pd.DataFrame.std, pd.DataFrame.skew, pd.DataFrame.kurt])
    # drop 'biomarker_type' multiindex column and add as prefix to columns
    out.columns = out.columns.droplevel(0)
    out = out.add_prefix("{}_".format(biomarker_type))
    out.columns.name = "biomarker"
    return out


def slope(data: pd.DataFrame, sample_idx: Union[Tuple[int, int], Sequence[int]],
          biomarker_type: Optional[Union[str, Sequence[str]]] = "cortisol",
          saliva_times: Optional[Sequence[int]] = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    _check_data_format(data)
    saliva_times = _get_saliva_times(data, saliva_times, remove_s0=False)
    _check_saliva_times(saliva_times)

    if biomarker_type not in data:
        raise ValueError("No `{}` columns in data!".format(biomarker_type))

    # ensure list
    sample_idx = list(sample_idx)

    if len(sample_idx) != 2:
        raise ValueError("Exactly 2 indices needed for computing slope. Got {} indices.".format(len(sample_idx)))

    if isinstance(biomarker_type, list):
        dict_result = {}
        for biomarker in biomarker_type:
            biomarker_cols = [biomarker]
            if 'time' in data:
                biomarker_cols = ['time'] + biomarker_cols
            dict_result[biomarker] = slope(data[biomarker_cols], sample_idx=sample_idx, biomarker_type=biomarker_type,
                                           saliva_times=saliva_times)

    data = data[[biomarker_type]].unstack()

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

    out = pd.DataFrame(np.diff(data.iloc[:, sample_idx]) / np.diff(saliva_times[..., sample_idx]), index=data.index,
                       columns=['{}_slope{}{}'.format(biomarker_type, *sample_idx)])
    out.columns.name = "biomarker"
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
        if saliva_times.ndim <= 2:
            saliva_times = saliva_times[..., 1:]
        else:
            raise ValueError("`saliva_times` has invalid dimensions: {}".format(saliva_times.ndim))

    return saliva_times
