from pathlib import Path
from typing import Optional, Sequence, Union, Tuple, Dict, Literal

import numpy as np
import pandas as pd
import pytz

from biopsykit._types import path_t
from biopsykit.utils.time import tz, utc

COUNTER_INCONSISTENCY_HANDLING = Literal["raise", "warn", "ignore"]


def load_dataset_nilspod(
    file_path: Optional[path_t] = None,
    dataset: Optional["Dataset"] = None,
    datastreams: Optional[Sequence[str]] = None,
    handle_counter_inconsistency: Optional[COUNTER_INCONSISTENCY_HANDLING] = "raise",
    legacy_support: Optional[str] = "resolve",
    timezone: Optional[Union[pytz.timezone, str]] = tz,
) -> Tuple[pd.DataFrame, int]:
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
    >>> from biopsykit.io.nilspod import load_dataset_nilspod
    >>> # Option 1: Import data by passing file name
    >>> file_path = "./<filename-of-nilspod-data>.bin"
    >>> # load dataset with all datastreams
    >>> df, fs = load_dataset_nilspod(file_path=file_path)
    >>> # load only ECG data of dataset
    >>> df, fs = load_dataset_nilspod(file_path=file_path, datastreams=['ecg'])
    >>>
    >>> # Option 2: Import data by passing Dataset object imported from NilsPodLib (in this example, only acceleration data)
    >>> from nilspodlib import Dataset
    >>> dataset = Dataset.from_bin_file("<filename>.bin")
    >>> df, fs = load_dataset_nilspod(dataset=dataset, datastreams=['acc'])
    """
    from nilspodlib import Dataset
    import warnings

    if file_path:
        dataset = Dataset.from_bin_file(file_path, legacy_support=legacy_support)
    if isinstance(timezone, str):
        # convert to pytz object
        timezone = pytz.timezone(timezone)

    idxs_corrupted = np.where(np.diff(dataset.counter) < 1)[0]

    # check if only last sample is corrupted and cut last sample
    if len(idxs_corrupted) == 1 and (idxs_corrupted == len(dataset.counter) - 2):
        dataset.cut(start=0, stop=idxs_corrupted[0], inplace=True)
    elif len(idxs_corrupted) > 1:
        if handle_counter_inconsistency == "raise":
            raise ValueError(
                "Error loading dataset. Counter not monotonously increasing!"
            )
        elif handle_counter_inconsistency == "warn":
            warnings.warn(
                "Counter not monotonously increasing. This might indicate that the dataset is corrupted or "
                "that the dataset was recorded as part of a synchronized session and might need to be loaded "
                "using `biopsykit.io.load_synced_session_nilspod()`. "
                "Check the counter of the DataFrame manually!"
            )

    # convert dataset to dataframe and localize timestamp
    df = (
        dataset.data_as_df(datastreams, index="utc_datetime")
        .tz_localize(tz=utc)
        .tz_convert(tz=timezone)
    )
    df.index.name = "time"
    return df, int(dataset.info.sampling_rate_hz)


def load_synced_session_nilspod(
    folder_path: path_t,
    datastreams: Optional[Sequence[str]] = None,
    handle_counter_inconsistency: Optional[COUNTER_INCONSISTENCY_HANDLING] = "raise",
    legacy_support: Optional[str] = "resolve",
    timezone: Optional[Union[pytz.timezone, str]] = tz,
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
            raise ValueError(
                "Error loading session. Counter not monotonously increasing!"
            )
        elif handle_counter_inconsistency == "warn":
            warnings.warn(
                "Counter not monotonously increasing. This might indicate that the session is corrupted. "
                "Check the counter of the DataFrame manually!"
            )

    # convert dataset to dataframe and localize timestamp
    df = (
        session.data_as_df(datastreams, index="utc_datetime", concat_df=True)
        .tz_localize(tz=utc)
        .tz_convert(tz=timezone)
    )
    df.index.name = "time"
    if len(set(session.info.sampling_rate_hz)) > 1:
        fs = tuple([int(s) for s in session.info.sampling_rate_hz])
    else:
        fs = int(session.info.sampling_rate_hz[0])
    return df, fs


def load_csv_nilspod(
    file_path: path_t = None,
    datastreams: Optional[Sequence[str]] = None,
    timezone: Optional[Union[pytz.timezone, str]] = tz,
) -> Tuple[pd.DataFrame, int]:
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
    start_time = (
        pd.to_datetime(start_time, format="%Y%m%d_%H%M%S").to_datetime64().astype(int)
    )
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


def load_folder_nilspod(
    folder_path: path_t,
    phase_names: Optional[Sequence[str]] = None,
    datastreams: Optional[Sequence[str]] = None,
    legacy_support: Optional[str] = "resolve",
    timezone: Optional[Union[pytz.timezone, str]] = tz,
) -> Tuple[Dict[str, pd.DataFrame], int]:
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
    import biopsykit.io.nilspod    >>> import biopsykit as bp
        >>> from biopsykit.io.nilspod import load_folder_nilspod
        >>> folder_path = "./nilspod"
        >>> # load all datasets from the selected folder with all datastreams
        >>> dataset_dict, fs = load_folder_nilspod(folder_path)
        >>> # load only ECG data of all datasets from the selected folder
        >>> dataset_dict, fs = load_folder_nilspod(folder_path, datastreams=['ecg'])
        >>> # load all datasets from the selected folder with correspondng phase names
        >>> dataset_dict, fs = load_folder_nilspod(folder_path, phase_names=['VP01','VP02','VP03'])
    """
    # ensure pathlib
    folder_path = Path(folder_path)
    # look for all NilsPod binary files in the folder
    dataset_list = list(sorted(folder_path.glob("*.bin")))
    if phase_names is None:
        phase_names = ["Part{}".format(i) for i in range(len(dataset_list))]

    if len(phase_names) != len(dataset_list):
        raise ValueError(
            "Number of phases does not match number of datasets in the folder!"
        )

    dataset_dict = {
        phase: load_dataset_nilspod(
            file_path=dataset_path,
            datastreams=datastreams,
            legacy_support=legacy_support,
            timezone=timezone,
        )
        for phase, dataset_path in zip(phase_names, dataset_list)
    }
    # assume equal sampling rates for all datasets in folder => take sampling rate from first dataset
    sampling_rate = list(dataset_dict.values())[0].info.sampling_rate_hz
    return dataset_dict, sampling_rate


def check_nilspod_dataset_corrupted(dataset: "Dataset") -> bool:
    return np.where(np.diff(dataset.counter) != 1.0)[0].size != 0


def get_nilspod_dataset_corrupted_info(dataset: "Dataset", file_path: path_t) -> Dict:
    import re

    nilspod_file_pattern = r"NilsPodX-\w{4}_(.*?).bin"
    # ensure pathlib
    file_path = Path(file_path)

    keys = ["name", "percent_corrupt", "condition"]
    dict_res = dict.fromkeys(keys)
    if not check_nilspod_dataset_corrupted(dataset):
        dict_res["condition"] = "fine"
        return dict_res

    idx_diff = np.diff(dataset.counter)
    idx_corrupt = np.where(idx_diff != 1.0)[0]
    percent_corrupt = (len(idx_corrupt) / len(idx_diff)) * 100.0
    condition = "parts"
    if percent_corrupt > 90.0:
        condition = "lost"
    elif percent_corrupt < 50.0:
        if (idx_corrupt[0] / len(idx_corrupt)) < 0.30:
            condition = "start_only"
        elif (idx_corrupt[0] / len(idx_corrupt)) > 0.70:
            condition = "end_only"

    dict_res["name"] = re.search(nilspod_file_pattern, file_path.name).group(1)
    dict_res["percent_corrupt"] = percent_corrupt
    dict_res["condition"] = condition
    return dict_res
