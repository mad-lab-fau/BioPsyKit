import warnings
from typing import Optional, Sequence, Tuple, Union, Dict

import pandas as pd
import numpy as np

from biopsykit.saliva.utils import _check_data_format, _check_saliva_times, _get_saliva_times


def max_increase(data: pd.DataFrame, biomarker_type: Optional[Union[str, Sequence[str]]] = "cortisol",
                 remove_s0: Optional[bool] = False,
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
    data = data[[biomarker_type]].unstack(level='sample')

    max_inc = (data.iloc[:, 1:].max(axis=1) - data.iloc[:, 0])
    if percent:
        max_inc = 100.0 * max_inc / np.abs(data.iloc[:, 0])

    out = pd.DataFrame(max_inc, columns=[
        "{}_max_inc_percent".format(biomarker_type) if percent else "{}_max_inc".format(biomarker_type)],
                       index=max_inc.index)
    out.columns.name = "biomarker"
    return out


def auc(data: pd.DataFrame, biomarker_type: Optional[Union[str, Sequence[str]]] = "cortisol",
        remove_s0: Optional[bool] = False,
        compute_auc_post: Optional[bool] = False,
        saliva_times: Optional[Union[np.ndarray, Sequence[int], str]] = None) -> Union[
    pd.DataFrame, Dict[str, pd.DataFrame]]:
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
    data = data[[biomarker_type]].unstack(level='sample')

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
