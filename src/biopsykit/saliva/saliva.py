"""Functions for processing saliva data and computing established features (AUC, slope, maximum increase, ...)."""
import warnings
from typing import Optional, Sequence, Tuple, Union, Dict, List

import pandas as pd
import numpy as np

from biopsykit.saliva.utils import (
    _check_sample_times,
    _get_sample_times,
    _get_saliva_idx_labels,
    _remove_s0,
    _get_group_cols,
)
from biopsykit.utils.datatype_helper import (
    SalivaRawDataFrame,
    is_saliva_raw_dataframe,
    is_saliva_feature_dataframe,
    SalivaFeatureDataFrame,
    is_saliva_mean_se_dataframe,
    SalivaMeanSeDataFrame,
    _SalivaFeatureDataFrame,
    _SalivaMeanSeDataFrame,
)
from biopsykit.utils.exceptions import DataFrameTransformationError
from biopsykit.utils.functions import se


def max_value(
    data: SalivaRawDataFrame,
    saliva_type: Optional[Union[str, Sequence[str]]] = "cortisol",
    remove_s0: Optional[bool] = False,
) -> Union[SalivaFeatureDataFrame, Dict[str, SalivaFeatureDataFrame]]:
    """Compute maximum value.

    The output feature name will be ``max_val``, preceded by the name of the saliva type to allow better
    conversion into long-format later on (if desired). So e.g., for cortisol, it will be: ``cortisol_max_val``.

    Parameters
    ----------
    data : :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`
        saliva data in `SalivaRawDataFrame` format
    saliva_type : str or list of str
        saliva type or list of saliva types to compute features on
    remove_s0 : bool, optional
        whether to remove the first saliva sample for computing maximum or not. Default: ``False``

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.SalivaFeatureDataFrame` or dict of such
        dataframe containing the computed features, or a dict of such if ``saliva_type`` is a list

    Raises
    ------
    :exc:`~biopsykit.utils.exceptions.ValidationError`
        if ``data`` is not a :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`

    """
    # check input
    is_saliva_raw_dataframe(data, saliva_type)

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
        columns=["{}_max_val".format(saliva_type)],
        index=max_val.index,
    )
    out.columns.name = "saliva_feature"

    # check output
    is_saliva_feature_dataframe(out, saliva_type)
    return out


def initial_value(
    data: SalivaRawDataFrame,
    saliva_type: Optional[Union[str, Sequence[str]]] = "cortisol",
    remove_s0: Optional[bool] = False,
) -> Union[SalivaFeatureDataFrame, Dict[str, SalivaFeatureDataFrame]]:
    """Compute initial saliva sample.

    The output feature name will be ``ini_val``, preceded by the name of the saliva type to allow better
    conversion into long-format later on (if desired). So e.g., for cortisol, it will be: ``cortisol_ini_val``.

    Parameters
    ----------
    data : :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`
        saliva data in `SalivaRawDataFrame` format
    saliva_type : str or list of str
        saliva type or list of saliva types to compute features on
    remove_s0 : bool, optional
        whether to remove the first saliva sample for computing initial value or not. Default: ``False``

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.SalivaFeatureDataFrame` or dict of such
        dataframe containing the computed features, or a dict of such if ``saliva_type`` is a list

    Raises
    ------
    :exc:`~biopsykit.utils.exceptions.ValidationError`
        if ``data`` is not a :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`

    """
    # check input
    is_saliva_raw_dataframe(data, saliva_type)

    if isinstance(saliva_type, list):
        dict_result = {}
        for saliva in saliva_type:
            saliva_col = [saliva]
            if "time" in data:
                saliva_col = saliva_col + ["time"]
            dict_result[saliva] = initial_value(
                data[saliva_col],
                saliva_type=saliva,
                remove_s0=remove_s0,
            )
        return dict_result

    if remove_s0:
        # We have a S0 sample => drop it
        data = _remove_s0(data)

    data = data[[saliva_type]].unstack(level="sample")

    ini_val = data.iloc[:, 0]

    out = pd.DataFrame(
        ini_val.values,
        columns=["{}_ini_val".format(saliva_type)],
        index=ini_val.index,
    )
    out.columns.name = "saliva_feature"

    # check output
    is_saliva_feature_dataframe(out, saliva_type)
    return out


def max_increase(
    data: SalivaRawDataFrame,
    saliva_type: Optional[Union[str, Sequence[str]]] = "cortisol",
    remove_s0: Optional[bool] = False,
    percent: Optional[bool] = False,
) -> Union[SalivaFeatureDataFrame, Dict[str, SalivaFeatureDataFrame]]:
    """Compute maximum increase between first saliva sample and all others.

    The maximum increase (`max_inc`) is defined as the difference between the `first` sample and the maximum of
    all `subsequent` samples.

    If the first sample should be excluded from computation, e.g., because the first sample was just collected for
    controlling against high initial saliva levels, ``remove_s0`` needs to set to ``True``.

    The output is either the absolute increase or the relative increase to the first sample in percent
    (if ``percent`` is ``True``).

    The output feature name will be ``max_inc``, preceded by the name of the saliva type to allow better
    conversion into long-format later on (if desired). So e.g., for cortisol, it will be: ``cortisol_max_inc``.

    Parameters
    ----------
    data : :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`
        saliva data in `SalivaRawDataFrame` format
    saliva_type : str or list of str
        saliva type or list of saliva types to compute features on
    remove_s0 : bool, optional
        whether to exclude the first saliva sample from computing `max_inc` or not. Default: ``False``
    percent : bool, optional
        whether to compute ``max_inc`` in percent (i.e., relative increase) or not. Default: ``False``

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.SalivaFeatureDataFrame` or dict of such
        dataframe containing the computed features, or a dict of such if ``saliva_type`` is a list

    Raises
    ------
    :exc:`~biopsykit.utils.exceptions.ValidationError`
        if ``data`` is not a :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`

    """
    # check input
    is_saliva_raw_dataframe(data, saliva_type)

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
        data = _remove_s0(data)

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
    is_saliva_feature_dataframe(out, saliva_type)
    return out


def auc(
    data: SalivaRawDataFrame,
    saliva_type: Optional[Union[str, Sequence[str]]] = "cortisol",
    remove_s0: Optional[bool] = False,
    compute_auc_post: Optional[bool] = False,
    sample_times: Optional[Union[np.ndarray, Sequence[int], str]] = None,
) -> Union[SalivaFeatureDataFrame, Dict[str, SalivaFeatureDataFrame]]:
    r"""Compute area-under-the-curve (AUC) for saliva samples.

    The area-under-the-curve is computed according to Pruessner et al. (2003) using the trapezoidal rule
    (:func:`numpy.trapz`). To compute an AUC the saliva time points are required in minutes. They can either be part of
    the :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame` (`time` column) or can be supplied as extra
    parameter (``sample_times``).

    Pruessner defined two types of AUC, which are computed by default:

    * AUC with respect to `ground` (:math:`AUC_{G}`), and
    * AUC with respect to the first sample, i.e., AUC with respect to `increase` (:math:`AUC_{I}`)

    If the first sample should be excluded from computation, e.g., because the first sample was just collected for
    controlling against high initial saliva levels, ``remove_s0`` needs to set to ``True``.

    If saliva samples were collected during an acute stress task :math:`AUC_{I}` can additionally be computed
    only for the saliva values *after* the stressor by setting ``compute_auc_post`` to ``True``.

    .. note::
        For a *pre/post* stress scenario *post*-stress saliva samples are indicated by time points :math:`t \geq 0`,
        saliva sampled collected *before* start of the stressor are indicated by time points :math:`t < 0`.
        This means that a saliva sample collected at time :math:`t = 0` is defined as *right after stressor*.

    The feature names will be ``auc_g``, ``auc_i`` (and ``auc_i_post`` if ``compute_auc_post`` is ``True``),
    preceded by the name of the saliva type to allow better conversion into long-format later on (if desired).
    So e.g., for cortisol, it will be: ``cortisol_auc_g``.

    Parameters
    ----------
    data : :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`
        saliva data in `SalivaRawDataFrame` format
    saliva_type : str or list of str
        saliva type or list of saliva types to compute features on
    remove_s0 : bool, optional
        whether to exclude the first saliva sample from computing ``auc`` or not. Default: ``False``
    compute_auc_post : bool, optional
        whether to additionally compute :math:`AUC_I` only for saliva samples *post* stressor.
        Saliva samples *post* stressor are defined as all samples with non-negative ``sample_times``.
        Default: ``False``
    sample_times: :any:`numpy.ndarray` or list, optional
        Saliva sampling times (corresponding to x-axis values for computing AUC). By default
        (``sample_times`` is ``None``) sample times are expected to be part of the dataframe (in the `time` column).
        Alternatively, sample times can be specified by passing a list or a numpy array to this argument.
        If ``sample_times`` is a 1D array, it is assumed that saliva times are the same for all subjects.
        Then, ``sample_times`` needs to have the shape (n_samples,).
        If ``sample_times`` is a 2D array, it is assumed that saliva times are individual for all subjects.
        Then, ``saliva_times`` needs to have the shape (n_subjects, n_samples).

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.SalivaFeatureDataFrame` or dict of such
        dataframe containing the computed features, or a dict of such if ``saliva_type`` is a list

    Raises
    ------
    :exc:`~biopsykit.utils.exceptions.ValidationError`
        if ``data`` is not a :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`

    References
    ----------
    Pruessner, J. C., Kirschbaum, C., Meinlschmid, G., & Hellhammer, D. H. (2003).
    Two formulas for computation of the area under the curve represent measures of total hormone concentration
    versus time-dependent change. Psychoneuroendocrinology, 28(7), 916–931.
    https://doi.org/10.1016/S0306-4530(02)00108-7

    """
    # check input
    is_saliva_raw_dataframe(data, saliva_type)

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
                sample_times=sample_times,
            )
        return dict_result

    sample_times = _get_sample_times(data, saliva_type, sample_times, remove_s0)
    _check_sample_times(sample_times)

    if remove_s0:
        # We have a S0 sample => drop it
        data = _remove_s0(data)

    data = data[[saliva_type]].unstack(level="sample")

    auc_data = {
        "auc_g": np.trapz(data, sample_times),
        "auc_i": np.trapz(data.sub(data.iloc[:, 0], axis=0), sample_times),
    }

    if compute_auc_post:
        auc_data = _auc_compute_auc_post(data, auc_data, sample_times)

    out = pd.DataFrame(auc_data, index=data.index).add_prefix("{}_".format(saliva_type))
    out.columns.name = "saliva_feature"

    # check output
    is_saliva_feature_dataframe(out, saliva_type)

    return _SalivaFeatureDataFrame(out)


def _auc_compute_auc_post(
    data: SalivaRawDataFrame, auc_data: Dict[str, np.ndarray], sample_times: np.ndarray
) -> Dict[str, np.ndarray]:
    idxs_post = None
    if sample_times.ndim == 1:
        idxs_post = np.where(sample_times >= 0)[0]
    elif sample_times.ndim == 2:
        warnings.warn(
            "Not computing `auc_i_post` values because this is only implemented if `saliva_times` "
            "are the same for all subjects."
        )
    if idxs_post is not None:
        data_post = data.iloc[:, idxs_post]
        auc_data["auc_i_post"] = np.trapz(data_post.sub(data_post.iloc[:, 0], axis=0), sample_times[idxs_post])
    return auc_data


def slope(
    data: SalivaRawDataFrame,
    sample_labels: Optional[Union[Tuple, Sequence]] = None,
    sample_idx: Optional[Union[Tuple[int, int], Sequence[int]]] = None,
    saliva_type: Optional[Union[str, Sequence[str]]] = "cortisol",
    sample_times: Optional[Sequence[int]] = None,
) -> Union[SalivaFeatureDataFrame, Dict[str, SalivaFeatureDataFrame]]:
    """Compute the slope between two saliva samples.

    The samples to compute the slope can either be specified by `index` (parameter `sample_idx`) [0, num_of_samples-1]
    or by `label` (parameter `sample_idx`).

    The output feature name for the slope between saliva samples with labels `label1` and `label2` will be
    ``slope<label1><label2>``, preceded by the name of the saliva type to allow better conversion into
    long-format later on (if desired). So e.g., for cortisol, it will be: ``cortisol_slopeS1S2``.

    Parameters
    ----------
    data : :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`
        saliva data in `SalivaRawDataFrame` format
    saliva_type : str or list of str
        saliva type or list of saliva types to compute features on
    sample_labels : list or tuple
        pair of saliva sample labels to compute slope between.
        Labels correspond to the names in the `sample` column of the dataframe.
        An error will the raised if not exactly 2 samples are specified.
    sample_idx : list or tuple
        pair of saliva sample indices to compute slope between.
        An error will the raised if not exactly 2 sample are specified
    sample_times: :any:`numpy.ndarray` or list, optional
        Saliva sampling times (corresponding to x-axis values for computing slope). By default
        (``sample_times`` is ``None``) sample times are expected to be part of the dataframe (in the `time` column).
        Alternatively, sample times can be specified by passing a list or a numpy array to this argument.
        If ``sample_times`` is a 1D array, it is assumed that saliva times are the same for all subjects.
        Then, ``sample_times`` needs to have the shape (n_samples,).
        If ``sample_times`` is a 2D array, it is assumed that saliva times are individual for all subjects.
        Then, ``saliva_times`` needs to have the shape (n_subjects, n_samples).

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.SalivaFeatureDataFrame` or dict of such
        dataframe containing the computed features, or a dict of such if ``saliva_type`` is a list

    Raises
    ------
    IndexError
        if invalid `sample_labels` or `sample_idx` is provided
    :exc:`~biopsykit.utils.exceptions.ValidationError`
        if ``data`` is not a :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`

    """
    # check input
    is_saliva_raw_dataframe(data, saliva_type)

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
                sample_times=sample_times,
            )
        return dict_result

    sample_times = _get_sample_times(data, saliva_type, sample_times)
    _check_sample_times(sample_times)

    data = data[[saliva_type]].unstack()

    sample_labels, sample_idx = _get_saliva_idx_labels(
        data[saliva_type].columns, sample_labels=sample_labels, sample_idx=sample_idx
    )

    out = pd.DataFrame(
        np.diff(data.iloc[:, sample_idx]) / np.diff(sample_times[..., sample_idx]),
        index=data.index,
        columns=["{}_slope{}{}".format(saliva_type, *sample_labels)],
    )
    out.columns.name = "saliva_feature"

    # check output
    is_saliva_feature_dataframe(out, saliva_type)

    return _SalivaFeatureDataFrame(out)


def standard_features(
    data: pd.DataFrame,
    saliva_type: Optional[Union[str, Sequence[str]]] = "cortisol",
    group_cols: Optional[Union[str, Sequence[str]]] = None,
    keep_index: Optional[bool] = True,
) -> Union[SalivaFeatureDataFrame, Dict[str, SalivaFeatureDataFrame]]:
    """Compute a set of `standard features` on saliva data.

    The following list of features is computed:

    * ``argmax``: Argument (=index) of the maximum value
    * ``mean``: Mean value
    * ``std``: Standard deviation
    * ``skew``: Skewness
    * ``kurt``: Kurtosis

    For all features the built-in pandas functions (e.g. :meth:`pandas.DataFrame.mean`) will be used,
    except for ``argmax``, which will use numpy's function (:func:`numpy.argmax`). The functions will be applied on the
    dataframe using the `aggregate` functions from pandas (:meth:`pandas.DataFrame.agg`).

    The output feature names will be ``argmax``, ``mean``, ``std``, ``skew``, ``kurt``, preceded by the name of the
    saliva type to allow better conversion into long-format later on (if desired).
    So e.g., for cortisol, it will be: ``cortisol_argmax``.

    Parameters
    ----------
    data : :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`
        saliva data in `SalivaRawDataFrame` format
    saliva_type : str or list of str
        saliva type or list of saliva types to compute features on
    group_cols: str or list of str, optional
        columns to group on before applying the aggregate function. If ``group_cols`` is ``None`` (the default),
        data will be grouped on by all columns except the `sample` column. Usually, data wants to be grouped by
        `subject` (followed by `condition`, `day`, `night`, etc., if applicable).
    keep_index : bool, optional
        whether to try keeping the old index or use the new index returned by the groupby-aggregate-function.
        Keeping the old index is e.g. useful if the dataframe has a multiindex with several levels, but grouping is
        only performed on a subset of these levels. Default: ``True``

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.SalivaFeatureDataFrame` or dict of such
        dataframe containing the computed features, or a dict of such if ``saliva_type`` is a list

    Raises
    ------
    :exc:`~biopsykit.utils.exceptions.ValidationError`
        if ``data`` is not a :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`
    :exc:`~biopsykit.utils.exceptions.DataFrameTransformationError`
        if ``keep_index`` is ``True``, but applying the old index fails

    """
    # check input
    is_saliva_raw_dataframe(data, saliva_type)

    if isinstance(saliva_type, list):
        dict_result = {}
        for saliva in saliva_type:
            saliva_col = [saliva]
            if "time" in data:
                saliva_col = saliva_col + ["time"]
            dict_result[saliva] = standard_features(data[saliva_col], saliva_type=saliva)
        return dict_result

    group_cols = _get_group_cols(data, group_cols, "sample", "standard_features")

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
    is_saliva_feature_dataframe(out, saliva_type)

    return _SalivaFeatureDataFrame(out)


def mean_se(
    data: SalivaRawDataFrame,
    saliva_type: Optional[Union[str, Sequence[str]]] = "cortisol",
    group_cols: Optional[Union[str, List[str]]] = None,
    remove_s0: Optional[bool] = False,
) -> Union[SalivaMeanSeDataFrame, Dict[str, SalivaMeanSeDataFrame]]:
    """Compute mean and standard error per saliva sample.

    Parameters
    ----------
    data : :class:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`
        saliva data in `SalivaRawDataFrame` format
    saliva_type : str or list of str
        saliva type or list of saliva types to compute features on
    group_cols: str or list of str, optional
        columns to group on before computing mean and se. If ``group_cols`` is ``None`` (the default),
        data will be grouped on by all columns except the `sample` column. Usually, data wants to be grouped by
        `subject` (followed by `condition`, `day`, `night`, etc., if applicable).
    remove_s0 : bool, optional
        whether to exclude the first saliva sample from computing mean and standard error or not. Default: ``False``

    Returns
    -------
    :class:`~biopsykit.utils.datatype_helper.SalivaMeanSeDataFrame`
        dataframe with mean and standard error per saliva sample or a dict of such if ``saliva_type`` is a list

    Raises
    ------
    :exc:`~biopsykit.utils.exceptions.ValidationError`
        if ``data`` is not a :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`


    """
    # check input
    is_saliva_raw_dataframe(data, saliva_type)

    if isinstance(saliva_type, list):
        dict_result = {}
        for biomarker in saliva_type:
            biomarker_cols = [biomarker]
            if "time" in data:
                biomarker_cols = ["time"] + biomarker_cols
            dict_result[biomarker] = mean_se(data[biomarker_cols], saliva_type=biomarker, remove_s0=remove_s0)
        return dict_result

    if remove_s0:
        # We have a S0 sample => drop it
        data = _remove_s0(data)

    group_cols = _get_group_cols(data, group_cols, "subject", "mean_se")
    _mean_se_assert_group_cols(data, group_cols)

    if "time" in data.columns and "time" not in group_cols:
        # add 'time' column to grouper if it's in the data and wasn't added yet because otherwise
        # we would loose this column
        group_cols = group_cols + ["time"]

    data_mean_se = data.groupby(group_cols).agg([np.mean, se])[saliva_type]
    is_saliva_mean_se_dataframe(data_mean_se)

    return _SalivaMeanSeDataFrame(data_mean_se)


def _mean_se_assert_group_cols(data: pd.DataFrame, group_cols: Sequence[str]):
    if group_cols == list(data.index.names):
        # if data should be grouped by *all* index levels we can not compute an aggregation
        raise DataFrameTransformationError(
            "Cannot compute mean and standard error on data because *all* index "
            "columns were selected as group columns, so each sample value would be one group of its own!"
        )
    if "sample" not in group_cols:
        raise DataFrameTransformationError(
            "Error computing mean and standard error on data: 'sample' index level needs to be added as group column!"
        )
