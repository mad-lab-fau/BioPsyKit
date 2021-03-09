import datetime
from pathlib import Path
from typing import Optional, Union, Sequence, Dict, Tuple, Literal

import numpy as np
import pandas as pd
import pytz

from biopsykit.utils import path_t, utc, tz

COUNTER_INCONSISTENCY_HANDLING = Literal['raise', 'warn', 'ignore']


def load_dataset_nilspod(file_path: Optional[path_t] = None,
                         dataset: Optional['Dataset'] = None,
                         datastreams: Optional[Sequence[str]] = None,
                         handle_counter_inconsistency: Optional[COUNTER_INCONSISTENCY_HANDLING] = 'raise',
                         legacy_support: Optional[str] = 'resolve',
                         timezone: Optional[Union[pytz.timezone, str]] = tz) -> Tuple[pd.DataFrame, int]:
    """
    Converts a file recorded by NilsPod into a dataframe.

    You can either pass a Dataset object obtained from `nilspodlib` or directly pass the path to the file to load 
    and convert the file at once.

    Parameters
    ----------
    file_path : str or path, optional
        path to dataset object to converted
    dataset : Dataset
        Dataset object to convert
    datastreams : list of str, optional
        list of datastreams of the Dataset if only specific ones should be included or `None` to load all datastreams.
        Datastreams that are not part of the current dataset will be silently ignored.
    handle_counter_inconsistency : str, optional
         how to handle if counter of dataset is not monotonously increasing, which might be an indicator for a
         corrupted dataset. `raise` to raise an error, `warn` to issue a warning but still return a dataframe,
         `ignore` to ignore the counter check result
    legacy_support : str, optional
        This indicates how to deal with older NilsPod firmware versions.
        If `error`: An error is raised, if an unsupported version is detected.
        If `warn`: A warning is raised, but the file is parsed without modification
        If `resolve`: A legacy conversion is performed to load old files. If no suitable conversion is found,
        an error is raised. See the `legacy` package and the README of `nilspodlib` to learn more about available
        conversions.
        Default: `resolve`
    timezone : str or pytz.timezone, optional
            timezone of the acquired data to convert, either as string of as pytz object (default: 'Europe/Berlin')

    Returns
    -------
    tuple
        tuple of pandas dataframe with sensor data and sampling rate

    Examples
    --------
    >>> import biopsykit as bp
    >>> # Option 1: Import data by passing file name
    >>> file_path = "./<filename-of-nilspod-data>.bin"
    >>> # load dataset with all datastreams
    >>> df, fs = bp.io.load_dataset_nilspod(file_path=file_path)
    >>> # load only ECG data of dataset
    >>> df, fs = bp.io.load_dataset_nilspod(file_path=file_path, datastreams=['ecg'])
    >>>
    >>> # Option 2: Import data by passing Dataset object imported from NilsPodLib (in this example, only acceleration data)
    >>> from nilspodlib import Dataset
    >>> dataset = Dataset.from_bin_file("<filename>.bin")
    >>> df, fs = bp.io.load_dataset_nilspod(dataset=dataset, datastreams=['acc'])
    """
    from nilspodlib import Dataset
    import warnings

    if file_path:
        dataset = Dataset.from_bin_file(file_path, legacy_support=legacy_support)
    if isinstance(timezone, str):
        # convert to pytz object
        timezone = pytz.timezone(timezone)

    if len(np.where(np.diff(dataset.counter) < 1)[0]) > 0:
        if handle_counter_inconsistency == "raise":
            raise ValueError("Error loading dataset. Counter not monotonously increasing!")
        elif handle_counter_inconsistency == "warn":
            warnings.warn("Counter not monotonously increasing. This might indicate that the dataset is corrupted or "
                          "that the dataset was recorded as part of a synchronized session and might need to be loaded "
                          "using `biopsykit.io.load_synced_session_nilspod()`. "
                          "Check the counter of the DataFrame manually!")

    # convert dataset to dataframe and localize timestamp
    df = dataset.data_as_df(datastreams, index="utc_datetime").tz_localize(tz=utc).tz_convert(tz=timezone)
    df.index.name = "time"
    return df, int(dataset.info.sampling_rate_hz)


def load_synced_session_nilspod(folder_path: path_t,
                                datastreams: Optional[Sequence[str]] = None,
                                handle_counter_inconsistency: Optional[COUNTER_INCONSISTENCY_HANDLING] = 'raise',
                                legacy_support: Optional[str] = 'resolve',
                                timezone: Optional[Union[pytz.timezone, str]] = tz
                                ) -> Tuple[pd.DataFrame, Union[int, Tuple[int, ...]]]:
    from nilspodlib import SyncedSession
    import warnings

    nilspod_files = list(sorted(folder_path.glob("*.bin")))
    if len(nilspod_files) == 0:
        raise ValueError("No NilsPod files found in directory!")

    session = SyncedSession.from_folder_path(folder_path, legacy_support=legacy_support)
    session.align_to_syncregion(inplace=True)

    if len(np.where(np.diff(session.counter) < 1)[0]) > 0:
        if handle_counter_inconsistency == "raise":
            raise ValueError("Error loading session. Counter not monotonously increasing!")
        elif handle_counter_inconsistency == "warn":
            warnings.warn("Counter not monotonously increasing. This might indicate that the session is corrupted. "
                          "Check the counter of the DataFrame manually!")

    # convert dataset to dataframe and localize timestamp
    df = session.data_as_df(datastreams, index="utc_datetime", concat_df=True).tz_localize(tz=utc).tz_convert(
        tz=timezone)
    df.index.name = "time"
    if len(set(session.info.sampling_rate_hz)) > 1:
        fs = tuple([int(s) for s in session.info.sampling_rate_hz])
    else:
        fs = int(session.info.sampling_rate_hz[0])
    return df, fs


def load_csv_nilspod(file_path: Optional[path_t] = None, datastreams: Optional[Sequence[str]] = None,
                     timezone: Optional[Union[pytz.timezone, str]] = tz) -> Tuple[pd.DataFrame, int]:
    """
    Converts a CSV file recorded by NilsPod into a dataframe.

    Parameters
    ----------
    file_path : str or path
        path to dataset object to load
    datastreams : list of str, optional
        list of datastreams of the Dataset if only specific ones should be included or `None` to load all datastreams.
        Datastreams that are not part of the current dataset will be silently ignored.
    timezone : str or pytz.timezone, optional
        timezone of the acquired data to convert, either as string of as pytz object (default: 'Europe/Berlin')

    Returns
    -------
    tuple
        tuple of pandas dataframe with sensor data and sampling rate

    See Also
    --------
    `biopsykit.io.load_dataset_nilspod()`

    """
    import re

    df = pd.read_csv(file_path, header=1, index_col="timestamp")
    header = pd.read_csv(file_path, header=None, nrows=1)

    # infer start time from filename
    start_time = re.findall(r"NilsPodX-[^\s]{4}_(.*?).csv", str(file_path.name))[0]
    start_time = pd.to_datetime(start_time, format="%Y%m%d_%H%M%S").to_datetime64().astype(int)
    # sampling rate is in second column of header
    sampling_rate = int(header.iloc[0, 1])
    # convert index to nanoseconds
    df.index = ((df.index / sampling_rate) * 1e9).astype(int)
    # add start time as offset and convert into datetime index
    df.index = pd.to_datetime(df.index + start_time)
    df.index.name = "time"

    df_filt = pd.DataFrame(index=df.index)
    if datastreams is None:
        df_filt = df
    else:
        # filter only desired datastreams
        for ds in datastreams:
            df_filt = df_filt.join(df.filter(like=ds))

    if isinstance(timezone, str):
        # convert to pytz object
        timezone = pytz.timezone(timezone)
    # localize timezone (is already in correct timezone since start time is inferred from file name)
    df_filt = df_filt.tz_localize(tz=timezone)
    return df_filt, sampling_rate


def load_folder_nilspod(folder_path: path_t, phase_names: Optional[Sequence[str]] = None,
                        datastreams: Optional[Sequence[str]] = None,
                        legacy_support: Optional[str] = 'resolve',
                        timezone: Optional[Union[pytz.timezone, str]] = tz) -> Tuple[Dict[str, pd.DataFrame], int]:
    """
    Loads all NilsPod datasets from one folder, converts them into dataframes and combines them into one dictionary.

    This function can for example be used when single session were recorded for different phases.

    Parameters
    ----------
    folder_path : str or path
        path to folder containing data
    phase_names: list, optional
        list of phase names corresponding to the files in the folder. Must match the number of recordings
    datastreams : list of str, optional
        list of datastreams of the Dataset if only specific ones should be included or `None` to load all datastreams.
        Datastreams that are not part of the current dataset will be silently ignored.
    legacy_support : str, optional
        This indicates how to deal with older NilsPod firmware versions.
        If `error`: An error is raised, if an unsupported version is detected.
        If `warn`: A warning is raised, but the file is parsed without modification
        If `resolve`: A legacy conversion is performed to load old files. If no suitable conversion is found,
        an error is raised. See the `legacy` package and the README of `nilspodlib` to learn more about available
        conversions.
        Default: `resolve`
    timezone : str or pytz.timezone, optional
            timezone of the acquired data to convert, either as string of as pytz object (default: 'Europe/Berlin')

    Returns
    -------
    tuple
        tuple of dictionary with phase names as keys and pandas dataframes with sensor data as values and sampling rate

    Raises
    ------
    ValueError
        if number of phases does not match the number of datasets in the folder

    Examples
    --------
    >>> import biopsykit as bp
    >>> folder_path = "./nilspod"
    >>> # load all datasets from the selected folder with all datastreams
    >>> dataset_dict, sampling_rate = bp.io.load_folder_nilspod(folder_path)
    >>> # load only ECG data of all datasets from the selected folder
    >>> dataset_dict, sampling_rate = bp.io.load_dataset_nilspod(folder_path, datastreams=['ecg'])
    >>> # load all datasets from the selected folder with correspondng phase names
    >>> dataset_dict, sampling_rate = bp.io.load_dataset_nilspod(folder_path, phase_names=['VP01','VP02','VP03'])
    """
    # ensure pathlib
    folder_path = Path(folder_path)
    # look for all NilsPod binary files in the folder
    dataset_list = list(sorted(folder_path.glob("*.bin")))
    if phase_names is None:
        phase_names = ["Part{}".format(i) for i in range(len(dataset_list))]

    if len(phase_names) != len(dataset_list):
        raise ValueError("Number of phases does not match number of datasets in the folder!")

    dataset_dict = {
        phase: load_dataset_nilspod(file_path=dataset_path, datastreams=datastreams, legacy_support=legacy_support,
                                    timezone=timezone) for
        phase, dataset_path in zip(phase_names, dataset_list)}
    # assume equal sampling rates for all datasets in folder => take sampling rate from first dataset
    sampling_rate = list(dataset_dict.values())[0].info.sampling_rate_hz
    return dataset_dict, sampling_rate


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
    from biopsykit.questionnaires.utils import convert_nan
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


def load_stroop_inquisit_data(folder_path=str, cols: Optional[Sequence[str]] = None) -> Dict[str,pd.DataFrame]:
    """
    Loads the stroop test data from a folder and writes parameters like mean response time, number of correct answers,..
    into a Dictionary. The raw data needs to be as an .iqdat format in the path folder.

    Parameters
    ----------
    folder_path : str
        path to the folder in which the stroop test data is kept
    cols : Sequence(str)
        column name of which data should be load into the dictionary
    Returns
    -------
    dict
        dictionary with parameters of the stroop test like number of correct answers, mean response time,...
    """
    dict_stroop = {}
    dict_stroop_subphase = {}
    subject = ""
    # ensure pathlib
    folder_path = Path(folder_path)
    # look for all Inquisit files in the folder
    dataset_list = list(sorted(folder_path.glob("*.iqdat")))
    #iterate through data
    for data_path in dataset_list:

        df_stroop = pd.read_csv(data_path, sep='\t')

        if(subject != df_stroop['subject'][0]):
            dict_stroop_subphase = {}

        #set subject, stroop phase
        subject = df_stroop['subject'][0]
        subphase = 'Stroop' + str(df_stroop['sessionid'][0])[-1]
        df_mean = df_stroop.mean(axis=0).to_frame().T

        if cols:
            dict_stroop_subphase[subphase] = df_mean[cols]
        else:
            dict_stroop_subphase[subphase] = df_mean

        dict_stroop[subject] = dict_stroop_subphase

    return dict_stroop


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


def check_nilspod_dataset_corrupted(dataset: 'Dataset') -> bool:
    return np.where(np.diff(dataset.counter) != 1.0)[0].size != 0


def get_nilspod_dataset_corrupted_info(dataset: 'Dataset', file_path: path_t) -> Dict:
    import re
    nilspod_file_pattern = r"NilsPodX-\w{4}_(.*?).bin"
    # ensure pathlib
    file_path = Path(file_path)

    keys = ['name', 'percent_corrupt', 'condition']
    dict_res = dict.fromkeys(keys)
    if not check_nilspod_dataset_corrupted(dataset):
        dict_res['condition'] = 'fine'
        return dict_res

    idx_diff = np.diff(dataset.counter)
    idx_corrupt = np.where(idx_diff != 1.0)[0]
    percent_corrupt = ((len(idx_corrupt) / len(idx_diff)) * 100.0)
    condition = "parts"
    if percent_corrupt > 90.0:
        condition = "lost"
    elif percent_corrupt < 50.0:
        if (idx_corrupt[0] / len(idx_corrupt)) < 0.30:
            condition = "start_only"
        elif (idx_corrupt[0] / len(idx_corrupt)) > 0.70:
            condition = "end_only"

    dict_res['name'] = re.search(nilspod_file_pattern, file_path.name).group(1)
    dict_res['percent_corrupt'] = percent_corrupt
    dict_res['condition'] = condition
    return dict_res
