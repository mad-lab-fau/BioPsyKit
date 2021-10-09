"""Module providing interactive widgets to select and display log data from the *CARWatch App*."""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import pandas as pd
from typing_extensions import Literal, get_args

from biopsykit.utils._types import path_t

if TYPE_CHECKING:
    import ipywidgets.widgets

    from biopsykit.carwatch_logs import LogData

LOG_FILENAME_PATTERN = "logs_(.*?)"

INPUT_TYPES = Literal["file", "folder"]
VALUE_TYPES = Literal["file_path", "file_name", "folder_name", "subject_id"]


def log_file_subject_dropdown(
    base_path: path_t,
    input_type: Optional[INPUT_TYPES] = "file",
    value_type: Optional[VALUE_TYPES] = "file_path",
    callback: Optional[Callable] = None,
) -> ipywidgets.Dropdown:
    """Create dropdown widget to select log files from one subject.

    Parameters
    ----------
    base_path : :class:`pathlib.Path` or str
        base path to log files from all subjects. log files are expected to be either stored in one folder
        (``input_type`` == "file") or in subfolders per subject (``input_type`` == "folder")
    input_type : {"file", "folder"}, optional
        string specifying how log data is present: .csv files, all in one folder ("file") or .csv files, all in
        separate subfolders per subject ("folder")
    value_type : {"file_path", "file_name", "folder_name", "subject_id"}
        string specifying output format of selected data:
            * "file_path": callback returns absolute file path of selected log data
            * "file_name": callback returns file name of selected log data
            * "folder_name": callback returns folder name of selected log data
            * "subject_id": callback returns Subject ID of selected log data
    callback : function, optional
        function reference to be used as callback function or ``None`` for no callback. Default: ``None``

    Returns
    -------
    :class:`~ipywidgets.widgets.widget_selection.Dropdown`
        dropdown widget

    """
    try:
        import ipywidgets.widgets  # pylint:disable=import-outside-toplevel
    except ImportError as e:
        raise ImportError(
            "Creating widget failed because ipywidgets cannot be imported. Install it via 'pip install ipywidgets'."
        ) from e

    _log_file_subject_dropdown_check_input(input_type, value_type)

    # ensure pathlib
    base_path = Path(base_path)

    if input_type == "file":
        log_file_pattern = LOG_FILENAME_PATTERN + ".csv"
        log_file_list = list(sorted(base_path.glob("*.csv")))
        subject_list = [re.search(log_file_pattern, log_file.name).group(1) for log_file in log_file_list]
    else:
        log_file_list = [
            folder for folder in base_path.glob("*") if folder.is_dir() and not folder.name.startswith(".")
        ]
        subject_list = [folder.name for folder in log_file_list]

    option_list = _log_file_subject_dropdown_get_option_list(subject_list, log_file_list, value_type)

    widget = ipywidgets.Dropdown(options=option_list, description="Subject ID")
    if callback:
        widget.observe(callback, names="value")
    return widget


def _log_file_subject_dropdown_check_input(input_type: str, value_type: str):
    if input_type not in get_args(INPUT_TYPES):
        raise ValueError("Invalid input_type! Expected one of {}, got {}.".format(INPUT_TYPES, input_type))

    if value_type not in get_args(VALUE_TYPES):
        raise ValueError("Invalid value_type! Expected one of {}, got {}.".format(VALUE_TYPES, value_type))


def _log_file_subject_dropdown_get_option_list(subject_list: List[str], log_file_list: List[Path], value_type: str):
    option_list: List[Tuple[str, Optional[Union[Path, str]]]] = [("Select Subject", None)]
    if value_type == "file_path":
        option_list = option_list + list(zip(subject_list, log_file_list))
    elif value_type in ["folder_name", "subject_id"]:
        option_list = option_list + list(zip(subject_list, subject_list))
    else:
        option_list = option_list + list(zip(subject_list, [log_file.name for log_file in log_file_list]))

    return option_list


def action_dropdown_widget(
    data: Union["LogData", pd.DataFrame], callback: Optional[Callable] = None
) -> "ipywidgets.Dropdown":
    """Create dropdown widget to filter log data by a specific action.

    Parameters
    ----------
    data : :class:`~biopsykit.carwatch_logs.log_data.LogData` or :class:`~pandas.DataFrame`
        log data as ``LogData`` object or as dataframe
    callback : function, optional
        function reference to be used as callback function or ``None`` for no callback. Default: ``None``

    Returns
    -------
    :class:`~ipywidgets.widgets.widget_selection.Dropdown`
        dropdown widget

    """
    from biopsykit.carwatch_logs import LogData  # pylint:disable=import-outside-toplevel

    try:
        import ipywidgets.widgets  # pylint:disable=import-outside-toplevel
    except ImportError as e:
        raise ImportError(
            "Creating widget failed because ipywidgets cannot be imported. Install it via 'pip install ipywidgets'."
        ) from e

    options = [("Select Action", None)]

    if isinstance(data, LogData):
        data = data.data

    avail_actions = list(data["action"].unique())
    options = options + list(zip(avail_actions, avail_actions))

    widget = ipywidgets.Dropdown(options=options, description="Action")
    if callback:
        widget.observe(callback, names="value")
    return widget


def day_dropdown_widget(
    data: Union["LogData", pd.DataFrame], callback: Optional[Callable] = None
) -> "ipywidgets.Dropdown":
    """Create dropdown widget to filter log data by a specific day.

    Parameters
    ----------
    data : :class:`~biopsykit.carwatch_logs.log_data.LogData` or :class:`~pandas.DataFrame`
        log data as ``LogData`` object or as dataframe
    callback : function, optional
        function reference to be used as callback function or ``None`` for no callback. Default: ``None``

    Returns
    -------
    :class:`~ipywidgets.widgets.widget_selection.Dropdown`
        dropdown widget

    """
    from biopsykit.carwatch_logs import LogData  # pylint:disable=import-outside-toplevel

    try:
        import ipywidgets.widgets  # pylint:disable=import-outside-toplevel
    except ImportError as e:
        raise ImportError(
            "Creating widget failed because ipywidgets cannot be imported. Install it via 'pip install ipywidgets'."
        ) from e

    options = [("Select Day", None)]

    if isinstance(data, LogData):
        data = data.data

    dates = data.index.normalize().unique()
    dates = [str(date.date()) for date in dates]

    options = options + list(zip(dates, dates))
    widget = ipywidgets.Dropdown(options=options, description="Day")
    if callback:
        widget.observe(callback, names="value")
    return widget
