"""Functions to compute established features (area-under-the-curve, slope, maximum increase, ...) from saliva data."""
import warnings
from typing import Optional, Sequence, Tuple, Union, Dict

import pandas as pd
import numpy as np

from biopsykit.saliva.utils import (
    _check_saliva_times,
    _get_saliva_times,
    _get_saliva_idx_labels,
)
from biopsykit.utils.datatype_helper import (
    SalivaRawDataFrame,
    is_raw_saliva_dataframe,
    is_feature_saliva_dataframe,
    SalivaFeatureDataFrame,
)
from biopsykit.utils.exceptions import DataFrameTransformationError


def max_value(
    data: SalivaRawDataFrame,
    saliva_type: Optional[Union[str, Sequence[str]]] = "cortisol",
    remove_s0: Optional[bool] = False,
):
    """Compute maximum value.

    The feature name of the variable will be ``max_val``, preceded by the name of the saliva type to allow better
    conversion into long-format later on (if desired) (so e.g., for cortisol, it will be: ``cortisol_max_val``).

    Parameters
    ----------
    data : :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`
        saliva data in :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame` format
    saliva_type : str or list of str
        saliva type or list of saliva types to compute features on
    remove_s0 : bool, optional
        whether to remove the first saliva sample for computing maximum or not. Default: ``False``

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.SalivaFeatureDataFrame`
    or dict of :obj:`~biopsykit.utils.datatype_helper.SalivaFeatureDataFrame`
        :obj:`~biopsykit.utils.datatype_helper.SalivaFeatureDataFrame` containing the computed features, or a dict of
        such if ``saliva_type`` is a list

    Raises
    ------
    ValidationError
        if ``data`` is not a SalivaRawDataFrame

    """
    # check input
    is_raw_saliva_dataframe(data, saliva_type)

    if isinstance(saliva_type, list):
        dict_result = {}
        for saliva in saliva_type:
            saliva_col = [saliva]
            if "time" in data:
                saliva_col = saliva_col + ["time"]
            dict_result[saliva] = max_value(
                data[saliva_col],
                saliva_type=saliva,
                remove_s0=remove_s0,
            )
        return dict_result

    if remove_s0:
        # We have a S0 sample => drop it
        data = data.drop(0, level="sample", errors="ignore")
        data = data.drop("0", level="sample", errors="ignore")
        data = data.drop("S0", level="sample", errors="ignore")

    data = data[[saliva_type]].unstack(level="sample")

    max_val = data.max(axis=1)

    out = pd.DataFrame(
        max_val,
        columns=["{}_max_val".format(saliva_type, saliva_type[0])],
        index=max_val.index,
    )
    out.columns.name = "saliva_feature"

    # check output
    is_feature_saliva_dataframe(out, saliva_type)
    return out


def max_increase(
    data: SalivaRawDataFrame,
    saliva_type: Optional[Union[str, Sequence[str]]] = "cortisol",
    remove_s0: Optional[bool] = False,
    percent: Optional[bool] = False,
) -> Union[SalivaFeatureDataFrame, Dict[str, SalivaFeatureDataFrame]]:
    """Compute maximum increase between first saliva sample and all others.

    The maximum increase (`max_inc`) is defined as the difference between the `first` sample
    (or the second sample, if ``remove_s0`` is ``True``, e.g., because the first sample is just for
    controlling for high initial saliva levels) and the maximum of the `subsequent` samples. The output is either
    absolute increase or in percent as relative increase to the first sample (``percent`` is ``True``).

    Parameters
    ----------
    data : :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`
        saliva data in :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame` format
    saliva_type : str or list of str
        saliva type or list of saliva types to compute features on
    remove_s0 : bool, optional
        whether to remove the first saliva sample for computing `max_inc` or not. Default: ``False``
    percent :
        whether to compute `max_inc` in percent (i.e., relative increase) or not. Default: ``False``

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.SalivaFeatureDataFrame`
    or dict of :obj:`~biopsykit.utils.datatype_helper.SalivaFeatureDataFrame`
        :obj:`~biopsykit.utils.datatype_helper.SalivaFeatureDataFrame` containing the computed features, or a dict of
        such if ``saliva_type`` is a list

    Raises
    ------
    ValidationError
        if ``data`` is not a SalivaRawDataFrame

    """
    # check input
    is_raw_saliva_dataframe(data, saliva_type)

    if isinstance(saliva_type, list):
        dict_result = {}
        for saliva in saliva_type:
            saliva_col = [saliva]
            if "time" in data:
                saliva_col = saliva_col + ["time"]
            dict_result[saliva] = max_increase(
                data[saliva_col],
                saliva_type=saliva,
                remove_s0=remove_s0,
                percent=percent,
            )
        return dict_result

    if remove_s0:
        # We have a S0 sample => drop it
        data = data.drop(0, level="sample", errors="ignore")
        data = data.drop("0", level="sample", errors="ignore")
        data = data.drop("S0", level="sample", errors="ignore")

    data = data[[saliva_type]].unstack(level="sample")

    max_inc = data.iloc[:, 1:].max(axis=1) - data.iloc[:, 0]
    if percent:
        max_inc = 100.0 * max_inc / np.abs(data.iloc[:, 0])

    out = pd.DataFrame(
        max_inc,
        columns=["{}_max_inc_percent".format(saliva_type) if percent else "{}_max_inc".format(saliva_type)],
        index=max_inc.index,
    )
    out.columns.name = "saliva_feature"

    # check output
    is_feature_saliva_dataframe(out, saliva_type)
    return out


def auc(
    data: SalivaRawDataFrame,
    saliva_type: Optional[Union[str, Sequence[str]]] = "cortisol",
    remove_s0: Optional[bool] = False,
    compute_auc_post: Optional[bool] = False,
    saliva_times: Optional[Union[np.ndarray, Sequence[int], str]] = None,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    # TODO add documentation; IMPORTANT: saliva_time '0' is defined as "right before stress" (0 min of stress)
    # => auc_post means all saliva times after beginning of stress (>= 0)

    # check input
    is_raw_saliva_dataframe(data, saliva_type)
    saliva_times = _get_saliva_times(data, saliva_times, remove_s0)
    _check_saliva_times(saliva_times)

    if isinstance(saliva_type, list):
        dict_result = {}
        for saliva in saliva_type:
            saliva_col = [saliva]
            if "time" in data:
                saliva_col = saliva_col + ["time"]
            dict_result[saliva] = auc(
                data[saliva_col],
                saliva_type=saliva,
                remove_s0=remove_s0,
                saliva_times=saliva_times,
            )
        return dict_result

    if remove_s0:
        # We have a S0 sample => drop it
        data = data.drop(0, level="sample", errors="ignore")
        data = data.drop("0", level="sample", errors="ignore")
        data = data.drop("S0", level="sample", errors="ignore")

    data = data[[saliva_type]].unstack(level="sample")

    auc_data = {
        "auc_g": np.trapz(data, saliva_times),
        "auc_i": np.trapz(data.sub(data.iloc[:, 0], axis=0), saliva_times),
    }

    if compute_auc_post:
        idxs_post = None
        if saliva_times.ndim == 1:
            idxs_post = np.where(saliva_times > 0)[0]
        elif saliva_times.ndim == 2:
            warnings.warn(
                "Not computing `auc_i_post` values because this is only implemented if `saliva_times` "
                "are the same for all subjects."
            )
        if idxs_post is not None:
            data_post = data.iloc[:, idxs_post]
            auc_data["auc_i_post"] = np.trapz(data_post.sub(data_post.iloc[:, 0], axis=0), saliva_times[idxs_post])

    out = pd.DataFrame(auc_data, index=data.index).add_prefix("{}_".format(saliva_type))
    out.columns.name = "saliva_feature"

    # check output
    is_feature_saliva_dataframe(out, saliva_type)

    return out


def standard_features(
    data: pd.DataFrame,
    saliva_type: Optional[Union[str, Sequence[str]]] = "cortisol",
    group_cols: Optional[Union[str, Sequence[str]]] = None,
    keep_index: Optional[bool] = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:

    # check input
    is_raw_saliva_dataframe(data, saliva_type)

    if isinstance(saliva_type, list):
        dict_result = {}
        for saliva in saliva_type:
            saliva_col = [saliva]
            if "time" in data:
                saliva_col = saliva_col + ["time"]
            dict_result[saliva] = standard_features(data[saliva_col], saliva_type=saliva)
        return dict_result

    if isinstance(group_cols, str):
        # ensure list
        group_cols = [group_cols]

    if group_cols is None:
        # group by all available index levels
        group_cols = list(data.index.names)
        group_cols.remove("sample")

    out = (
        data[[saliva_type]]
        .groupby(group_cols)
        .agg(
            [
                np.argmax,
                pd.DataFrame.mean,
                pd.DataFrame.std,
                pd.DataFrame.skew,
                pd.DataFrame.kurt,
            ],
        )
    )
    if keep_index:
        try:
            out.index = data.unstack(level="sample").index
        except ValueError as e:
            raise DataFrameTransformationError(
                "DataFrame transformation failed: Unable to keep old dataframe index because index does not match with "
                "output data shape, possibly because 'groupby' recuded the index. "
                "Consider setting 'keep_index' to 'False'. "
                "The exact error was:\n\n{}".format(str(e))
            ) from e

    # drop 'saliva_type' multiindex column and add as prefix to columns to ensure consistent naming with
    # the other saliva functions
    out.columns = out.columns.droplevel(0)
    out = out.add_prefix("{}_".format(saliva_type))
    out.columns.name = "saliva_feature"

    # check output
    is_feature_saliva_dataframe(out, saliva_type)

    return out


def slope(
    data: SalivaRawDataFrame,
    sample_labels: Optional[Union[Tuple, Sequence]] = None,
    sample_idx: Optional[Union[Tuple[int, int], Sequence[int]]] = None,
    saliva_type: Optional[Union[str, Sequence[str]]] = "cortisol",
    saliva_times: Optional[Sequence[int]] = None,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:

    # check input
    is_raw_saliva_dataframe(data, saliva_type)

    saliva_times = _get_saliva_times(data, saliva_times, remove_s0=False)
    _check_saliva_times(saliva_times)

    if sample_idx is None and sample_labels is None:
        raise IndexError("Either `sample_labels` or `sample_idx` must be supplied as parameter!")

    if sample_idx is not None and sample_labels is not None:
        raise IndexError("Either `sample_labels` or `sample_idx` must be supplied as parameter, not both!")

    if isinstance(saliva_type, list):
        dict_result = {}
        for saliva in saliva_type:
            saliva_col = [saliva]
            if "time" in data:
                saliva_col = saliva_col + ["time"]
            dict_result[saliva] = slope(
                data[saliva_col],
                sample_labels=sample_labels,
                sample_idx=sample_idx,
                saliva_type=saliva,
                saliva_times=saliva_times,
            )
        return dict_result

    data = data[[saliva_type]].unstack()

    sample_labels, sample_idx = _get_saliva_idx_labels(
        data[saliva_type].columns, sample_labels=sample_labels, sample_idx=sample_idx
    )

    out = pd.DataFrame(
        np.diff(data.iloc[:, sample_idx]) / np.diff(saliva_times[..., sample_idx]),
        index=data.index,
        columns=["{}_slope{}{}".format(saliva_type, *sample_labels)],
    )
    out.columns.name = "saliva_feature"

    # check output
    is_feature_saliva_dataframe(out, saliva_type)

    return out
