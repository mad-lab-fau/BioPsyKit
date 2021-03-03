import datetime
from pathlib import Path
from typing import Optional, Union, Sequence, Dict, Literal, List

import pandas as pd
import pytz

from biopsykit.utils import path_t

COUNTER_INCONSISTENCY_HANDLING = Literal['raise', 'warn', 'ignore']


def load_time_log(file_path: path_t, index_cols: Optional[Union[str, Sequence[str], Dict[str, str]]] = None,
                  phase_cols: Optional[Union[Sequence[str], Dict[str, str]]] = None) -> pd.DataFrame:
    """
    Loads a 'time log file', i.e. a file where time information about start and stop times of recordings or recording
    phases are stored.

    Parameters
    ----------
    file_path : str or path
        path to time log file, either Excel or csv file
    index_cols : str list, optional
        column name (or list of column names) that should be used for dataframe index or ``None`` for no index.
        Default: ``None``
    phase_cols : list, optional
        list of column names that contain time log information or ``None`` to use all columns. Default: ``None``

    Returns
    -------
    pd.DataFrame
        pandas dataframe with time log information

    Raises
    ------
    ValueError
        if file format is none of [.xls, .xlsx, .csv]


    >>> import biopsykit as bp
    >>> file_path = "./timelog.csv"
    >>> # load time log file into a pandas dataframe
    >>> df_time_log = bp.io.load_time_log(file_path)
    >>> # load time log file into a pandas dataframe and specify the 'subject_id' column in the time log file to be the index of the dataframe
    >>> df_time_log = bp.io.load_time_log(file_path, index_cols='subject_id')
    >>> # load time log file into a pandas dataframe and specify the columns 'Phase1' 'Phase2' and 'Phase3' in the time log file to be the used for extracting time information
    >>> df_time_log = bp.io.load_time_log(file_path, phase_cols=['Phase1', 'Phase2', 'Phase3'])
    """
    # ensure pathlib
    file_path = Path(file_path)
    if file_path.suffix in ['.xls', '.xlsx']:
        df_time_log = pd.read_excel(file_path)
    elif file_path.suffix in ['.csv']:
        df_time_log = pd.read_csv(file_path)
    else:
        raise ValueError("Unrecognized file format {}!".format(file_path.suffix))

    new_index_cols = None
    if isinstance(index_cols, str):
        index_cols = [index_cols]
    elif isinstance(index_cols, dict):
        new_index_cols = list(index_cols.values())
        index_cols = list(index_cols.keys())

    new_phase_cols = None
    if isinstance(phase_cols, dict):
        new_phase_cols = phase_cols
        phase_cols = list(phase_cols.keys())

    if index_cols:
        df_time_log.set_index(index_cols, inplace=True)
    if new_index_cols:
        df_time_log.index.rename(new_index_cols, inplace=True)

    if phase_cols:
        df_time_log = df_time_log.loc[:, phase_cols]
    if new_phase_cols:
        df_time_log.rename(columns=new_phase_cols, inplace=True)
    return df_time_log


def load_subject_condition_list(file_path: path_t, subject_col: Optional[str] = 'subject',
                                condition_col: Optional[str] = 'condition',
                                return_dict: Optional[bool] = True) -> Union[Dict, pd.DataFrame]:
    # enforce subject ID to be string
    df_cond = pd.read_csv(file_path, dtype={condition_col: str, subject_col: str})
    df_cond.set_index(subject_col, inplace=True)

    if return_dict:
        return df_cond.groupby(condition_col).groups
    else:
        return df_cond


def load_questionnaire_data(file_path: path_t,
                            index_cols: Optional[Union[str, Sequence[str]]] = None,
                            remove_nan_rows: Optional[bool] = True,
                            replace_missing_vals: Optional[bool] = True,
                            sheet_name: Optional[Union[str, int]] = 0) -> pd.DataFrame:
    from biopsykit.utils.dataframe_handling import convert_nan
    # ensure pathlib
    file_path = Path(file_path)
    if file_path.suffix == '.csv':
        data = pd.read_csv(file_path, index_col=index_cols)
    elif file_path.suffix in ('.xlsx', '.xls'):
        data = pd.read_excel(file_path, index_col=index_cols, sheet_name=sheet_name)
    else:
        raise ValueError("Invalid file type!")
    if remove_nan_rows:
        data = data.dropna(how='all')
    if replace_missing_vals:
        data = convert_nan(data)
    return data


def convert_time_log_datetime(time_log: pd.DataFrame, dataset: Optional['Dataset'] = None,
                              df: Optional[pd.DataFrame] = None, date: Optional[Union[str, 'datetime']] = None,
                              timezone: Optional[str] = "Europe/Berlin") -> pd.DataFrame:
    """
    Converts the time information of a time log pandas dataframe into datetime objects, i.e. adds the recording date
    to the time. Thus, either a NilsPodLib 'Dataset' or pandas DataFrame with DateTimeIndex must be supplied from which
    the recording date can be extracted or the date must explicitly be specified.

    Parameters
    ----------
    time_log : pd.DataFrame
        pandas dataframe with time log information
    dataset : Dataset, optional
        Dataset object to convert time log information into datetime
    df : pd.DataFrame, optional
    date : str or datatime, optional
        date to convert into time log into datetime
    timezone : str or pytz.timezone, optional
        timezone of the acquired data to convert, either as string of as pytz object (default: 'Europe/Berlin')

    Returns
    -------
    pd.DataFrame
        pandas dataframe with log time converted into datetime

    Raises
    ------
    ValueError
        if none of `dataset`, `df` and `date` are supplied as argument
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
            raise ValueError("Index of DataFrame must be DatetimeIndex!")
    if isinstance(date, str):
        # ensure datetime
        date = datetime.datetime(date)
    time_log = time_log.applymap(lambda x: pytz.timezone(timezone).localize(datetime.datetime.combine(date, x)))
    return time_log


def write_pandas_dict_excel(data_dict: Dict[str, pd.DataFrame], file_path: path_t,
                            index_col: Optional[bool] = True) -> None:
    """
    Writes a dictionary containing pandas dataframes to an Excel file.

    Parameters
    ----------
    data_dict : dict
        dict with pandas dataframes
    file_path : str or path
        filepath
    index_col : bool, optional
        ``True`` to include dataframe index in Excel export, ``False`` otherwise. Default: ``True``
    """
    # ensure pathlib
    file_path = Path(file_path)

    writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
    for key in data_dict:
        if isinstance(data_dict[key].index, pd.DatetimeIndex):
            # un-localize DateTimeIndex because Excel doesn't support timezone-aware dates
            data_dict[key].tz_localize(None).to_excel(writer, sheet_name=key, index=index_col)
        else:
            data_dict[key].to_excel(writer, sheet_name=key, index=index_col)
    writer.save()


def write_result_dict(result_dict: Dict[str, pd.DataFrame], file_path: path_t,
                      identifier_col: Optional[str] = "subject",
                      index_cols: Optional[List[str]] = ["phase"],
                      overwrite_file: Optional[bool] = False) -> None:
    """
    Saves dictionary with processing results (e.g. HR, HRV, RSA) of all subjects as csv file.

    Simply pass a dictionary with processing results. The keys in the dictionary should be the Subject IDs
    (or any other identifier), the values should be pandas dataframes. The resulting index can be specified by the
    `identifier_col` parameter.

    The dictionary will be concatenated to one large dataframe which will then be saved as csv file.

    *Notes*:
        * If a file with same name exists at the specified location, it is assumed that this is a result file from a
          previous run and the current results should be appended to this file.
          (To disable this behavior, set `overwrite_file` to ``False``).
        * Per default, it is assumed that the 'values' dataframes has a multi-index with columns ["phase"].
          The resulting dataframe would, thus, per default have the columns ["subject", "phase"].
          This can be changed by the parameter `index_cols`.

    Parameters
    ----------
    result_dict : dict
        Dictionary containing processing results for all subjects. The keys in the dictionary should be the Subject IDs
        (or any other identifier), the values should be pandas dataframes
    file_path : path, str
        path to file
    identifier_col : str, optional
        Name of the index in the concatenated dataframe. Default: "Subject_ID"
    index_cols : list of str, optional
        List of index columns of the single dataframes in the dictionary. Not needed if `overwrite_file` is ``False``.
        Default: ["Phase", "Subphase"]
    overwrite_file : bool, optional
        ``True`` to overwrite the file if it already exists, ``False`` otherwise. Default: ``True``

    Examples
    --------
    >>>
    >>> from biopsykit.io import write_result_dict
    >>>
    >>> file_path = "./param_results.csv"
    >>>
    >>> dict_param_output = {
    >>> 'S01' : pd.DataFrame(), # dataframe from mist_param_subphases,
    >>> 'S02' : pd.DataFrame(),
    >>> # ...
    >>> }
    >>>
    >>> write_result_dict(dict_param_output, file_path=file_path,
    >>>                             identifier_col="subject", index_cols=["phase", "subphase"])
    """

    # ensure pathlib
    file_path = Path(file_path)

    # TODO check if index_cols is really needed?

    identifier_col = [identifier_col]

    if index_cols is None:
        index_cols = []

    df_result_concat = pd.concat(result_dict, names=identifier_col + index_cols)
    if file_path.exists() and not overwrite_file:
        # ensure that all identifier columns are read as str
        df_result_old = pd.read_csv(file_path, dtype={col: str for col in identifier_col})
        df_result_old.set_index(identifier_col + index_cols, inplace=True)
        df_result_concat = df_result_concat.combine_first(df_result_old).sort_index(level=0)
    df_result_concat.reset_index().to_csv(file_path, index=False)
