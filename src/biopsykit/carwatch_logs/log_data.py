"""Module providing classes and utility functions for handling log data from *CARWatch App*."""
import json
import warnings
from datetime import datetime
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from typing_extensions import Literal

from biopsykit.carwatch_logs import log_actions, log_extras
from biopsykit.utils.time import tz

subject_conditions: Dict[str, str] = {
    "UNDEFINED": "Undefined",
    "KNOWN_ALARM": "Known Alarm",
    "UNKNOWN_ALARM": "Unknown Alarm",
    "SPONTANEOUS": "Spontaneous Awakening",
}

smartphone_models: Dict[str, str] = {
    "Nexus 7": "Google Nexus 7",
    "HTC 10": "HTC 10",
    "ALE-L21": "Huawei P8 Lite",
    "VTR-L29": "Huawei P10",
    "VOG-L29": "Huawei P30 Pro",
    "FIG-LX1": "Huawei P Smart",
    "MEDION S5004": "MEDION S5004",
    "Moto G (4)": "Motorola Moto G4",
    "Moto G (5)": "Motorola Moto G5",
    "ONEPLUS A6013": "OnePlus 6T",
    "Redmi Note 7": "Redmi Note 7",
    "SM-G920F": "Samsung Galaxy S6",
    "SM-G930F": "Samsung Galaxy S7",
    "SM-G950F": "Samsung Galaxy S8",
    "SM-G973F": "Samsung Galaxy S10",
    "SM-G970F": "Samsung Galaxy S10e",
    "SM-A750FN": "Samsung Galaxy A7",
    "SM-A205F": "Samsung Galaxy A20",
    "SM-A520F": "Samsung Galaxy A5",
    "SM-A500FU": "Samsung Galaxy A5",
    "Mi A1": "Xiaomi Mi A1",
}


class LogDataInfo:
    """Class representing general log data information."""

    def __init__(
        self,
        subject_id: str,
        condition: str,
        log_days: Optional[Sequence[datetime]] = None,
    ):
        """Initialize a new ``LogDataInfo`` instance.

        Parameters
        ----------
        subject_id : str
            subject ID
        condition : str
            study condition of participant
        log_days : list of :class:`datetime.datetime`, optional
            list of dates during which log data was collected or ``None`` to leave empty. Default: ``None``

        """
        self.subject_id: str = subject_id
        self.condition: str = condition
        if log_days is None:
            log_days = []
        self.log_days: Sequence[datetime] = log_days

        self.app_metadata: Dict[str, Union[int, str]] = {
            log_extras.app_version_code: 10000,
            log_extras.app_version_name: "1.0.0",
        }
        self.phone_metadata: Dict[str, str] = {
            log_extras.brand: "",
            log_extras.manufacturer: "",
            log_extras.model: "",
            log_extras.version_sdk_level: 0,
            log_extras.version_security_patch: "",
            log_extras.version_release: "",
        }

    @property
    def app_version_code(self) -> int:
        """App version code.

        Returns
        -------
        int
            version code of CARWatch App

        """
        return self.app_metadata[log_extras.app_version_code]

    @property
    def app_version_name(self) -> str:
        """Return app version name.

        Returns
        -------
        str
            version name of CARWatch App

        """
        return self.app_metadata[log_extras.app_version_name]

    @property
    def model(self) -> str:
        """Return smartphone model.

        Returns
        -------
        str
            name of smartphone model or "n/a" if information is not available

        """
        return self.phone_metadata[log_extras.model] if self.phone_metadata else "n/a"

    @property
    def manufacturer(self) -> str:
        """Return smartphone manufacturer.

        Returns
        -------
        str
            name of smartphone manufacturer or "n/a" if information is not available

        """
        return self.phone_metadata[log_extras.manufacturer] if self.phone_metadata else "n/a"

    @property
    def android_version(self) -> int:
        """Return Android version.

        Returns
        -------
        int
            SDK version of Android version or 0 if information is not available

        """
        return self.phone_metadata[log_extras.version_sdk_level] if self.phone_metadata else 0


class LogData:
    """Class representing log data."""

    log_actions: Dict[str, Sequence[str]] = {
        log_actions.app_metadata: [
            log_extras.app_version_code,
            log_extras.app_version_name,
        ],
        log_actions.phone_metadata: [
            log_extras.brand,
            log_extras.manufacturer,
            log_extras.model,
            log_extras.version_sdk_level,
            log_extras.version_security_patch,
            log_extras.version_release,
        ],
        log_actions.subject_id_set: [
            log_extras.subject_id,
            log_extras.subject_condition,
        ],
        log_actions.alarm_set: [
            log_extras.alarm_id,
            log_extras.timestamp,
            log_extras.is_repeating,
            log_extras.is_hidden,
            log_extras.hidden_timestamp,
        ],
        log_actions.timer_set: [log_extras.alarm_id, log_extras.timestamp],
        log_actions.alarm_cancel: [log_extras.alarm_id],
        log_actions.alarm_ring: [log_extras.alarm_id, log_extras.saliva_id],
        log_actions.alarm_snooze: [
            log_extras.alarm_id,
            log_extras.snooze_duration,
            log_extras.source,
        ],
        log_actions.alarm_stop: [
            log_extras.alarm_id,
            log_extras.source,
            log_extras.saliva_id,
        ],
        log_actions.alarm_killall: [],
        log_actions.evening_salivette: [log_extras.alarm_id],
        log_actions.barcode_scan_init: [],
        log_actions.barcode_scanned: [
            log_extras.alarm_id,
            log_extras.saliva_id,
            log_extras.barcode_value,
        ],
        log_actions.invalid_barcode_scanned: [log_extras.barcode_value],
        log_actions.duplicate_barcode_scanned: [
            log_extras.barcode_value,
            log_extras.other_barcodes,
        ],
        log_actions.spontaneous_awakening: [log_extras.alarm_id],
        log_actions.lights_out: [],
        log_actions.day_finished: [log_extras.day_counter],
        log_actions.service_started: [],
        log_actions.service_stopped: [],
        log_actions.screen_off: [],
        log_actions.screen_on: [],
        log_actions.user_present: [],
        log_actions.phone_boot_init: [],
        log_actions.phone_boot_complete: [],
        # TODO add further log actions
    }

    def __init__(self, data: pd.DataFrame, error_handling: Optional[Literal["ignore", "warn"]] = "ignore"):
        """Initialize new ``LogData`` instance.

        Parameters
        ----------
        data : :class:`~pandas.DataFrame`
            log data as dataframe
        error_handling : {"ignore", "warn"}
            how to handle error when parse log data. ``error_handling`` can be one of the following:

            * "warn" to issue warning when no "Subject ID Set" action was found in the data (indicating that a
              participant did not correctly register itself for the study or that log data is corrupted)
            * "ignore" to ignore warning.

        """
        self.data: pd.DataFrame = data
        self.error_handling: str = error_handling
        self.selected_day = None
        self.selected_action = None
        self.info: LogDataInfo = self.extract_info()

    def extract_info(self) -> LogDataInfo:
        """Extract log data information.

        Returns
        -------
        :class:`~biopsykit.carwatch_logs.log_data.LogDataInfo`
            ``LogDataInfo`` object

        """
        # Subject Information
        subject_dict = get_extras_for_log(self, log_actions.subject_id_set)
        subject_id: str = ""
        condition: str = subject_conditions["UNDEFINED"]

        if subject_dict:
            subject_id = subject_dict[log_extras.subject_id]
            condition = subject_conditions.get(
                subject_dict[log_extras.subject_condition], subject_conditions["UNDEFINED"]
            )
        elif self.error_handling == "warn":
            warnings.warn("Action 'Subject ID Set' not found – Log Data may be invalid!")

        # App Metadata
        app_dict = get_extras_for_log(self, log_actions.app_metadata)
        # Phone Metadata
        phone_dict = get_extras_for_log(self, log_actions.phone_metadata)
        if log_extras.model in phone_dict and phone_dict[log_extras.model] in smartphone_models:
            phone_dict[log_extras.model] = smartphone_models[phone_dict[log_extras.model]]

        # Log Info
        log_days = np.array([ts.date() for ts in self.data.index.normalize().unique()])
        log_info = LogDataInfo(subject_id, condition, log_days)
        log_info.log_days = log_days
        log_info.phone_metadata = phone_dict
        if app_dict:
            log_info.app_metadata = app_dict
        return log_info

    def _ipython_display_(self):
        self.print_info()

    def print_info(self):
        """Display Markdown-formatted log data information."""
        try:
            from IPython.core.display import Markdown, display  # pylint:disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError(
                "Displaying LogData information failed because "
                "IPython cannot be imported. Install it via 'pip install ipython'."
            ) from e

        display(Markdown("Subject ID: **{}**".format(self.subject_id)))
        display(Markdown("Condition: **{}**".format(self.condition)))
        display(Markdown("App Version: **{}**".format(self.app_version)))
        display(Markdown("Android Version: **{}**".format(self.android_version)))
        display(Markdown("Phone: **{}**".format(self.model)))
        display(Markdown("Logging Days: **{} – {}**".format(str(self.start_date), str(self.end_date))))

    @property
    def subject_id(self) -> str:
        """Return Subject ID.

        Returns
        -------
        str
            Subject ID

        """
        return self.info.subject_id

    @property
    def condition(self) -> str:
        """Return study condition from log data.

        Returns
        -------
        str
            study condition from log data

        """
        return self.info.condition

    @property
    def android_version(self) -> int:
        """Return Android version.

        Returns
        -------
        int
            SDK version of Android version or 0 if information is not available

        """
        return self.info.android_version

    @property
    def app_version(self) -> str:
        """Return app version name.

        Returns
        -------
        str
            version name of CARWatch App

        """
        return self.info.app_version_name.split("_")[0]

    @property
    def manufacturer(self) -> str:
        """Return smartphone manufacturer.

        Returns
        -------
        str
            name of smartphone manufacturer or "n/a" if information is not available

        """
        return self.info.manufacturer

    @property
    def model(self) -> str:
        """Return smartphone model.

        Returns
        -------
        str
            name of smartphone model or "n/a" if information is not available

        """
        return self.info.model

    @property
    def finished_days(self) -> Sequence[datetime.date]:
        """Return list of days where CAR procedure was completely logged successfully.

        Returns
        -------
        list
            list of dates that were finished successfully

        """
        return get_logs_for_action(self, log_actions.day_finished).index

    @property
    def num_finished_days(self) -> int:
        """Return number of days where CAR procedure was completely logged successfully.

        Returns
        -------
        int
            number of successfully finished days

        """
        return len(self.finished_days)

    @property
    def log_dates(self) -> Sequence[datetime.date]:
        """Return list of all days with log data.

        Returns
        -------
        list
            list of dates that contain at least one log data event

        """
        return self.info.log_days

    @property
    def start_date(self) -> datetime.date:
        """Return start date of log data.

        Returns
        -------
        :class:`datetime.date`
            start date

        """
        if self.log_dates is not None and len(self.log_dates) > 0:
            return self.log_dates[0]
        return None

    @property
    def end_date(self) -> datetime.date:
        """Return end date of log data.

        Returns
        -------
        :class:`datetime.date`
            end date

        """
        if self.log_dates is not None and len(self.log_dates) > 0:
            return self.log_dates[-1]
        return None


def get_filtered_logs(log_data: LogData) -> pd.DataFrame:
    """Return filtered logs for selected action and selected day.

    Parameters
    ----------
    log_data : :class:`~biopsykit.carwatch_logs.log_data.LogData`
        log data

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with filtered log data

    """
    return get_logs_for_action(log_data, log_action=log_data.selected_action, selected_day=log_data.selected_day)


def get_logs_for_date(data: Union[LogData, pd.DataFrame], date: Union[str, datetime.date]) -> pd.DataFrame:
    """Filter log data for a specific date.

    Parameters
    ----------
    data : :class:`~biopsykit.carwatch_logs.log_data.LogData` or :class:`~pandas.DataFrame`
        log data as ``LogData`` object or as dataframe
    date : :class:`datetime.date` or str
        date to filter log data for

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with log data for specific date

    """
    if isinstance(data, LogData):
        data = data.data

    date = pd.Timestamp(date).tz_localize(tz)

    if date is pd.NaT:
        return data

    return data.loc[data.index.normalize() == date]


def split_nights(data: Union[LogData, pd.DataFrame], diff_hours: Optional[int] = 12) -> Sequence[pd.DataFrame]:
    """Split continuous log data into individual nights.

    This function splits log data into individual nights when two successive timestamps differ more than the threshold
    provided by ``diff_hours``.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        input log data
    diff_hours : int, optional
        minimum difference between two successive log data timestamps required to split data into individual nights

    Returns
    -------
    list
        list of dataframes with log data split into individual nights

    """
    if isinstance(data, LogData):
        data = data.data

    idx_split = np.where(np.diff(data.index, prepend=data.index[0]) > pd.Timedelta(diff_hours, "hours"))[0]
    list_nights = np.split(data, idx_split)
    return list_nights


def get_logs_for_action(
    data: Union[LogData, pd.DataFrame],
    log_action: str,
    selected_day: Optional[datetime.date] = None,
    rows: Optional[Union[str, int, Sequence[int]]] = None,
) -> Union[pd.DataFrame, pd.Series]:
    """Filter log data for a specific action.

    Parameters
    ----------
    data : :class:`~biopsykit.carwatch_logs.log_data.LogData` or :class:`~pandas.DataFrame`
        log data as ``LogData`` object or as dataframe
    log_action : :class:`datetime.date` or str
        action to filter log data for
    selected_day : :class:`datetime.date`, optional
        filter log data to only contain data from one selected day or ``None`` to include data from all days
    rows : str, int, or list of int, optional
        index label (or list of such) to slice filtered log data (e.g., only select the first action) or
        ``None`` to include all data. Default: ``None``

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with log data for specific action

    """
    if isinstance(data, LogData):
        data = data.data

    if selected_day is not None:
        data = get_logs_for_date(data, date=selected_day)

    if log_action is None:
        return data
    if log_action not in LogData.log_actions:
        return pd.DataFrame()

    if rows:
        actions = data[data["action"] == log_action].iloc[rows, :]
    else:
        actions = data[data["action"] == log_action]
    return actions


def get_extras_for_log(data: Union[LogData, pd.DataFrame], log_action: str) -> Dict[str, str]:
    """Extract log data extras from log data.

    Parameters
    ----------
    data : :class:`~biopsykit.carwatch_logs.log_data.LogData` or :class:`~pandas.DataFrame`
        log data as ``LogData`` object or as dataframe
    log_action : :class:`datetime.date` or str
        action to filter log data

    Returns
    -------
    dict
        dictionary with log extras for specific action

    """
    row = get_logs_for_action(data, log_action, rows=0)
    if row.empty:
        # warnings.warn("Log file has no action {}!".format(log_action))
        return {}

    return json.loads(row["extras"].iloc[0])
