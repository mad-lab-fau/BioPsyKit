"""Module containing different I/O functions to load time log data, subject condition lists, questionnaire data, etc."""

from pathlib import Path
from typing import Optional, Union, Sequence, Dict

import datetime
import pytz

import numpy as np
import pandas as pd
from nilspodlib import Dataset

from biopsykit.utils.exceptions import ValidationError
from biopsykit.utils.dataframe_handling import convert_nan
from biopsykit.utils.time import tz

from biopsykit.utils._datatype_validation_helper import _assert_file_extension, _assert_has_columns, _assert_is_dtype
from biopsykit.utils._types import path_t


def load_time_log(
    file_path: path_t,
    index_cols: Optional[Union[str, Sequence[str], Dict[str, str]]] = None,
    phase_cols: Optional[Union[Sequence[str], Dict[str, str]]] = None,
    continuous_time: Optional[bool] = True,
    **kwargs,
) -> pd.DataFrame:
    """Load time log information from file.

    This function can be used to load a file containing 'time logs', i.e.,
    information about start and stop times of recordings or recording phases per subject.

    Parameters
    ----------
    file_path : :any:`pathlib.path` or str
        path to time log file. Must either be an Excel or csv file
    index_cols : str, list of str or dict, optional
        specifies dataframe index. Can either be a string or a list strings to indicate column name(s) from the
        dataframe that should be used as index level(s), or ``None`` for no index.
        If the index levels of the time log dataframe should have different names than the columns in the file,
        a dict specifying the mapping (column_name : new_index_name) can be passed. Default: ``None``
    phase_cols : list of str or dict, optional
        list of column names that contain time log information or ``None`` to use all columns.
        If the column names of the time log dataframe should have different names than the columns in the file,
        a dict specifying the mapping (column_name : new_column_name) can be passed. Default: ``None``
    continuous_time: bool, optional
        flag indicating whether phases are continuous, i.e., whether the end of the previous phase is also the
        beginning of the next phase or not. Default: ``True``.
        If ``continuous_time`` is set to ``False``, the start and end columns of all phases must have the
        suffixes "_start" and "_end", respectively
    **kwargs
        Additional parameters that are passed to :func:`pandas.read_csv` or :func:`pandas.read_excel`

    Returns
    -------
    :class:`pandas.DataFrame`
        dataframe with time log information

    Raises
    ------
    :class:`~biopsykit.exceptions.FileExtensionError`
        if file format is none of [.xls, .xlsx, .csv]
    :class:`~biopsykit.exceptions.ValidationError`
        if ``continuous_time`` is ``False``, but 'start' and 'end' time columns of each phase do not match or
        none of these columns were found in the dataframe

    Examples
    --------
    >>> import biopsykit as bp
    >>> file_path = "./timelog.csv"
    >>> # Example 1:
    >>> # load time log file into a pandas dataframe
    >>> data = bp.io.load_time_log(file_path)
    >>> # Example 2:
    >>> # load time log file into a pandas dataframe and specify the 'subject_id' column
    >>> # in the time log file to be the index of the dataframe
    >>> data = bp.io.load_time_log(file_path, index_cols="subject_id")
    >>> # Example 3:
    >>> # load time log file into a pandas dataframe, specify the columns "subject_id" and "condition"
    >>> # to be the new index, and the columns 'Phase1' 'Phase2' and 'Phase3' to be the used
    >>> # for extracting time information
    >>> data = bp.io.load_time_log(
    >>>     file_path, index_cols=["subject_id", "condition"], phase_cols=["Phase1", "Phase2", "Phase3"]
    >>> )
    >>> # Example 4:
    >>> # load time log file into a pandas dataframe and specify the column "subject_id" as index, but the index name
    >>> # in the dataframe should be "subject"
    >>> data = bp.io.load_time_log(file_path,
    >>>     index_cols={"subject_id": "subject"},
    >>>     phase_cols=["Phase1", "Phase2", "Phase3"]
    >>> )

    """
    # ensure pathlib
    file_path = Path(file_path)

    _assert_file_extension(file_path, expected_extension=[".xls", ".xlsx", ".csv"])
    if file_path.suffix in [".xls", ".xlsx"]:
        # assert times in the excel sheet are imported as strings,
        # not to be automatically converted into datetime objects
        data = pd.read_excel(file_path, dtype=str, **kwargs)
    else:
        data = pd.read_csv(file_path, **kwargs)

    data = _apply_index_cols(data, index_cols=index_cols)

    new_phase_cols = None
    if isinstance(phase_cols, dict):
        new_phase_cols = phase_cols
        phase_cols = list(phase_cols.keys())

    if phase_cols:
        _assert_has_columns(data, [phase_cols])
        data = data.loc[:, phase_cols]
    if new_phase_cols:
        data = data.rename(columns=new_phase_cols)

    if not continuous_time:
        start_cols = np.squeeze(data.columns.str.extract(r"(\w+)_start").dropna().values)
        end_cols = np.squeeze(data.columns.str.extract(r"(\w+)_end").dropna().values)
        if start_cols.size == 0:
            raise ValidationError(
                "No 'start' and 'end' columns were found. "
                "Make sure that each phase has columns with 'start' and 'end' suffixes!"
            )
        if not np.array_equal(start_cols, end_cols):
            raise ValidationError("Not all phases have 'start' and 'end' columns!")

        if index_cols is None:
            index_cols = [s for s in ["subject", "condition"] if s in data.columns]
            data = data.set_index(index_cols)

        data = pd.wide_to_long(
            data.reset_index(),
            stubnames=start_cols,
            i=index_cols,
            j="time",
            sep="_",
            suffix="(start|end)",
        )
        print(data)

        # ensure that "start" is always before "end"
        data = data.reindex(["start", "end"], level=-1)
        # unstack start|end level
        data = data.unstack()

    print(data)
    print(data.values.flatten())
    for val in data.values.flatten():
        _assert_is_dtype(val, str)

    return data


def load_subject_condition_list(
    file_path: path_t,
    subject_col: Optional[Union[str, Dict[str, str]]] = "subject",
    condition_col: Optional[Union[str, Dict[str, str]]] = "condition",
    return_dict: Optional[bool] = False,
) -> Union[Dict, pd.DataFrame]:
    """Load subject condition assignment from file.

    This function can be used to load a file that contains the assignment of subject IDs to study conditions.

    Parameters
    ----------
    file_path : :any:`pathlib.path` or str
        path to time log file. Must either be an Excel or csv file
    subject_col : str or dict, optional
        name of column containing subject IDs, which will be the new index of the dataframe. If the name of the
        index level in the dataframe should have a different name than the column in the file, a dict specifying the
        mapping (column_name : new_index_name) can be passed.
        Default: ``subject``
    condition_col : str or dict, optional
        name of column containing condition assignments. If the name of the condition column in the dataframe should
        have a different name than the column in the file, a dict specifying the mapping
        (column_name : new_column_name) can be passed. Default: ``condition``
    return_dict : bool, optional
        whether to return a dict with subject IDs per condition (``True``) or a dataframe (``False``)

    Returns
    -------
        :class:`pandas.DataFrame` or dict
        a pandas dataframe with subject IDs and condition assignments (if ``return_dict`` is ``False``) or
        a dict with subject IDs per group (if ``return_dict`` is ``True``)

    Raises
    ------
    :class:`~biopsykit.exceptions.FileExtensionError`
        if file is not a csv file

    """
    # enforce subject ID to be string
    _assert_file_extension(file_path, expected_extension=".csv")

    data = pd.read_csv(file_path)
    if isinstance(subject_col, dict):
        data = data.rename(columns=subject_col)
        subject_col = list(subject_col.values())
    if isinstance(condition_col, dict):
        data = data.rename(columns=condition_col)

    data = data.set_index(subject_col)

    if return_dict:
        return data.groupby(condition_col).groups
    return data


def load_questionnaire_data(
    file_path: path_t,
    index_cols: Optional[Union[str, Sequence[str], Dict[str, str]]] = None,
    remove_nan_rows: Optional[bool] = True,
    replace_missing_vals: Optional[bool] = True,
    sheet_name: Optional[Union[str, int]] = 0,
    **kwargs,
) -> pd.DataFrame:
    """Load questionnaire data from file.

    Parameters
    ----------
    file_path : :any:`pathlib.path` or str
        path to time log file. Must either be an Excel or csv file
    index_cols : str, list of str or dict, optional
        specifies dataframe index. Can either be a string or a list strings to indicate column name(s) from the
        dataframe that should be used as index level(s), or ``None`` for no index.
        If the index levels of the time log dataframe should have different names than the columns in the file,
        a dict specifying the mapping (column_name : new_index_name) can be passed. Default: ``None``
    remove_nan_rows : bool, optional
        ``True`` to remove rows that only contain NaN values (except the index cols), ``False`` to keep NaN rows.
        Default: ``True``
    replace_missing_vals : bool, optional
        ``True`` to replace encoded "missing values" from software like SPSS (e.g. -77, -99, or -66)
         to "actual" missing values (NaN). Default: ``True``
    sheet_name : str or int, optional
        sheet_name identifier (str) or sheet_name index (int) if file is an Excel file.
        Default: 0 (i.e. first sheet in Excel file)

    Returns
    -------
    :class:`pandas.DataFrame`
        dataframe with imported questionnaire data

    Raises
    ------
    :class:`~biopsykit.exceptions.FileExtensionError`
        if file format is none of [.xls, .xlsx, .csv]

    """
    # ensure pathlib
    file_path = Path(file_path)

    _assert_file_extension(file_path, expected_extension=[".xls", ".xlsx", ".csv"])
    if file_path.suffix in [".xls", ".xlsx"]:
        data = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
    else:
        data = pd.read_csv(file_path, **kwargs)

    data = _apply_index_cols(data, index_cols=index_cols)

    if remove_nan_rows:
        data = data.dropna(how="all")
    if replace_missing_vals:
        data = convert_nan(data)
    return data


def load_stroop_inquisit_data(folder_path: path_t, cols: Optional[Sequence[str]] = None) -> Dict[str, pd.DataFrame]:
    """Load Inquisit data collected during "Stroop Test".

    Stroop Test reusults (mean response time, number of correct answers, etc.)
    are exported per Stroop phase and are stored in a common folder. This function loads all exported `.iqdat` files,
    transforms them into dataframes and combines them into a dictionary.

    Parameters
    ----------
    folder_path : :any:`pathlib.Path` or str
        path to the folder in which the Stroop test export files are stored
    cols : list of str, optional
        names of columns which should be imported and added to the dictionary

    Returns
    -------
    dict
        dictionary with Stroop Test parameters per Stroop Phase

    """
    dict_stroop = {}
    # ensure pathlib
    folder_path = Path(folder_path)
    # look for all Inquisit files in the folder
    dataset_list = list(sorted(folder_path.glob("*.iqdat")))
    subject = ""
    # iterate through data
    for data_path in dataset_list:
        df_stroop = pd.read_csv(data_path, sep="\t")
        if subject != df_stroop["subject"][0]:
            dict_stroop = {}
        # set subject, stroop phase
        subject = df_stroop["subject"][0]
        subphase = "Stroop{}".format(str(df_stroop["sessionid"][0])[-1])
        df_mean = df_stroop.mean(axis=0).to_frame().T

        if cols:
            dict_stroop[subphase] = df_mean[cols]
        else:
            dict_stroop[subphase] = df_mean

    return dict_stroop


def convert_time_log_datetime(
    time_log: pd.DataFrame,
    dataset: Optional[Dataset] = None,
    df: Optional[pd.DataFrame] = None,
    date: Optional[Union[str, datetime.datetime]] = None,
    timezone: Optional[Union[str, pytz.timezone]] = tz,
) -> pd.DataFrame:
    """Convert the time log information into datetime objects.

    This function converts time log information (containing only time, but no date)
    into datetime objects, thus, adds the `start date` of the recording. To specify the recording date,
    either a NilsPod :class:`nilspodlib.Dataset` or a pandas dataframe with a :class:`pandas.DateTimeIndex`
    must be supplied from which the recording date can be extracted.
    As an alternative, the date can be specified explicitly.

    Parameters
    ----------
    time_log : pd.DataFrame
        pandas dataframe with time log information
    dataset : :class:`nilspodlib.Dataset`, optional
        NilsPod Dataset object extract time and date information. Default: ``None``
    df : :class:`pandas.DataFrame`, optional
        dataframe with :class:`pandas.DateTimeIndex` to extract time and date information. Default: ``None``
    date : str or datetime, optional
        date used to convert time log information into datetime. Default: ``None``
    timezone : str or pytz.timezone, optional
        timezone of the acquired data to convert, either as string of as pytz object.
        Default: pytz.timezone('Europe/Berlin')

    Returns
    -------
    pd.DataFrame
        pandas dataframe with log time converted into datetime

    Raises
    ------
    ValueError
        if none of ``dataset``, ``df`` and ``date`` are supplied as argument,
        or if index of ``df`` is not a :class:`pandas.DatetimeIndex`

    """
    if dataset is None and date is None and df is None:
        raise ValueError("Either `dataset`, `df` or `date` must be supplied as argument!")

    if dataset is not None:
        date = dataset.info.utc_datetime_start.date()
    if df is not None:
        if isinstance(df.index, pd.DatetimeIndex):
            date = df.index.normalize().unique()[0]
            date = date.to_pydatetime()
        else:
            raise ValueError("Index of 'df' must be DatetimeIndex!")
    if isinstance(date, str):
        # ensure datetime
        date = datetime.datetime(date)
    time_log = time_log.applymap(lambda x: pytz.timezone(timezone).localize(datetime.datetime.combine(date, x)))
    return time_log


def write_pandas_dict_excel(
    data_dict: Dict[str, pd.DataFrame],
    file_path: path_t,
    index_col: Optional[bool] = True,
) -> None:
    """Write a dictionary with pandas dataframes to an Excel file.

    Parameters
    ----------
    data_dict : dict
        dictionary with pandas dataframes
    file_path : :any:`pathlib.Path` or str
        path to exported Excel file
    index_col : bool, optional
        ``True`` to include dataframe index in Excel export, ``False`` otherwise. Default: ``True``

    """
    # ensure pathlib
    file_path = Path(file_path)

    writer = pd.ExcelWriter(file_path, engine="xlsxwriter")  # pylint:disable=abstract-class-instantiated
    for key in data_dict:
        if isinstance(data_dict[key].index, pd.DatetimeIndex):
            # un-localize DateTimeIndex because Excel doesn't support timezone-aware dates
            data_dict[key].tz_localize(None).to_excel(writer, sheet_name=key, index=index_col)
        else:
            data_dict[key].to_excel(writer, sheet_name=key, index=index_col)
    writer.save()


def write_result_dict(
    result_dict: Dict[str, pd.DataFrame],
    file_path: path_t,
    index_name: Optional[str] = "subject",
) -> None:
    """Write dictionary with processing results (e.g. HR, HRV, RSA) to csv file.

    The keys in the dictionary should be the subject IDs (or any other identifier),´
    the values should be :class:`pandas.DataFrame`. The index level(s) of the exported dataframe can be specified
    by the ``index_col`` parameter.

    The dictionary will be concatenated to one large dataframe which will then be saved as csv file.

    Parameters
    ----------
    result_dict : dict
        Dictionary containing processing results for all subjects. The keys in the dictionary should be the Subject IDs
        (or any other identifier), the values should be pandas dataframes
    file_path : path, str
        path to file
    index_name : str, optional
        name of the index resulting from concatenting dataframes. Default: ``subject``

    Examples
    --------
    >>>
    >>> from biopsykit.io import write_result_dict
    >>>
    >>> file_path = "./param_results.csv"
    >>>
    >>> dict_param_output = {
    >>> 'S01' : pd.DataFrame(), # e.g., dataframe from mist_param_subphases,
    >>> 'S02' : pd.DataFrame(),
    >>> # ...
    >>> }
    >>>
    >>> write_result_dict(dict_param_output, file_path=file_path, index_name="subject")

    """
    # ensure pathlib
    file_path = Path(file_path)
    df_result_concat = pd.concat(result_dict, names=index_name)
    df_result_concat.to_csv(file_path)


def _apply_index_cols(
    data: pd.DataFrame, index_cols: Optional[Union[str, Sequence[str], Dict[str, str]]]
) -> pd.DataFrame:
    new_index_cols = None
    if isinstance(index_cols, str):
        index_cols = [index_cols]
    elif isinstance(index_cols, dict):
        new_index_cols = list(index_cols.values())
        index_cols = list(index_cols.keys())

    if index_cols:
        _assert_has_columns(data, [index_cols])
        data = data.set_index(index_cols)

    if new_index_cols:
        data.index = data.index.set_names(new_index_cols)

    return data
