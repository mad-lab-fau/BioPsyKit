from contextlib import contextmanager
from pathlib import Path
from unittest import TestCase

import pandas as pd
import pytest
from nilspodlib import Dataset
from pandas._testing import assert_index_equal

from biopsykit.io.nilspod import (
    check_nilspod_dataset_corrupted,
    get_nilspod_dataset_corrupted_info,
    load_csv_nilspod,
    load_dataset_nilspod,
    load_folder_nilspod,
    load_synced_session_nilspod,
)
from biopsykit.io.sleep import save_sleep_endpoints
from biopsykit.io.sleep_analyzer import (
    load_withings_sleep_analyzer_raw_file,
    load_withings_sleep_analyzer_raw_folder,
    load_withings_sleep_analyzer_summary,
)
from biopsykit.utils.exceptions import FileExtensionError, ValidationError

TEST_FILE_PATH = Path(__file__).parent.joinpath("../test_data/sleep_endpoints")


@contextmanager
def does_not_raise():
    yield


def sleep_endpoints_dataframe_correct():
    return pd.DataFrame(
        {
            "sleep_onset": pd.to_datetime("01.01.2021 00:00"),
            "wake_onset": pd.to_datetime("01.01.2021 08:00"),
            "total_sleep_duration": 8 * 60,
        },
        index=pd.DatetimeIndex(pd.to_datetime(["31.12.2020"]), name="date"),
    )


def sleep_endpoints_dataframe_additional_cols():
    return pd.DataFrame(
        {
            "sleep_onset": pd.to_datetime("01.01.2021 00:00"),
            "wake_onset": pd.to_datetime("01.01.2021 08:00"),
            "total_sleep_duration": 8 * 60,
            "major_rest_period_start": pd.to_datetime("31.12.2020 23:00"),
            "major_rest_period_end": pd.to_datetime("01.01.2021 08:30"),
        },
        index=pd.DatetimeIndex(pd.to_datetime(["31.12.2020"]), name="date"),
    )


def sleep_endpoints_dataframe_incorrect_index():
    return pd.DataFrame(
        {
            "sleep_onset": pd.to_datetime("01.01.2021 00:00"),
            "wake_onset": pd.to_datetime("01.01.2021 08:00"),
            "total_sleep_duration": 8 * 60,
        },
        index=range(0, 1),
    )


def sleep_endpoints_dataframe_missing_cols():
    return pd.DataFrame(
        {
            "sleep_onset": pd.to_datetime("01.01.2021 00:00"),
            "wake_onset": pd.to_datetime("01.01.2021 08:00"),
        },
        index=pd.DatetimeIndex(pd.to_datetime(["31.12.2020"]), name="date"),
    )


def sleep_endpoints_dict_correct():
    return {
        "date": pd.to_datetime("31.12.2020"),
        "sleep_onset": pd.to_datetime("01.01.2021 00:00"),
        "wake_onset": pd.to_datetime("01.01.2021 08:00"),
        "total_sleep_duration": 8 * 60,
    }


class TestIoSleep:
    @pytest.mark.parametrize(
        "file_path, sleep_endpoints, expected",
        [
            ("sleep_endpoints.csv", sleep_endpoints_dataframe_correct(), does_not_raise()),
            ("sleep_endpoints.csv", sleep_endpoints_dataframe_additional_cols(), does_not_raise()),
            ("sleep_endpoints.txt", sleep_endpoints_dataframe_correct(), pytest.raises(FileExtensionError)),
            ("sleep_endpoints.csv", sleep_endpoints_dict_correct(), pytest.raises(NotImplementedError)),
            ("sleep_endpoints.csv", sleep_endpoints_dataframe_incorrect_index(), pytest.raises(ValidationError)),
            ("sleep_endpoints.csv", sleep_endpoints_dataframe_missing_cols(), pytest.raises(ValidationError)),
        ],
    )
    def test_save_sleep_endpoints(self, file_path, sleep_endpoints, expected, tmp_path):
        with expected:
            save_sleep_endpoints(tmp_path.joinpath(file_path), sleep_endpoints)

    @pytest.mark.parametrize(
        "file_path, expected",
        [
            ("sleep_analyzer_summary.csv", does_not_raise()),
            ("sleep_analyzer_summary_wrong_column_names.csv", pytest.raises(ValidationError)),
        ],
    )
    def test_load_withings_sleep_analyzer_summary_raises(self, file_path, expected):
        with expected:
            load_withings_sleep_analyzer_summary(TEST_FILE_PATH.joinpath(file_path))

    @pytest.mark.parametrize(
        "file_path",
        [
            ("sleep_analyzer_summary.csv"),
        ],
    )
    def test_load_withings_sleep_analyzer_summary(self, file_path):
        data = load_withings_sleep_analyzer_summary(TEST_FILE_PATH.joinpath(file_path))
        assert_index_equal(
            data.index,
            pd.DatetimeIndex(
                pd.to_datetime(["10.10.2020", "10.12.2020"]).tz_localize("UTC").tz_convert("Europe/Berlin").normalize(),
                name="date",
            ),
        )

    @pytest.mark.parametrize(
        "file_path, data_source, timezone, expected",
        [
            ("raw_sleep-monitor_hr.csv", "heart_rate", None, does_not_raise()),
            ("raw_sleep-monitor_hr.csv", "heart_rate", "Europe/Berlin", does_not_raise()),
            ("raw_sleep-monitor_hr.csv", "hr", None, pytest.raises(ValueError)),
            ("raw_sleep-monitor_hr.csv", "hr", None, pytest.raises(ValueError)),
            ("sleep-monitor_hr_wrong_column_names.csv", "heart_rate", None, pytest.raises(ValidationError)),
        ],
    )
    def test_load_withings_sleep_analyzer_raw_file_raises(self, file_path, data_source, timezone, expected):
        with expected:
            load_withings_sleep_analyzer_raw_file(
                TEST_FILE_PATH.joinpath("sleep_analyzer").joinpath(file_path),
                data_source=data_source,
                timezone=timezone,
            )

    @pytest.mark.parametrize(
        "file_path, data_source",
        [
            ("raw_sleep-monitor_hr.csv", "heart_rate"),
        ],
    )
    def test_load_withings_sleep_analyzer_raw_file(self, file_path, data_source):
        data = load_withings_sleep_analyzer_raw_file(
            TEST_FILE_PATH.joinpath("sleep_analyzer").joinpath(file_path),
            data_source=data_source,
        )
        # TODO: add further checks to sleep analyzer import, such as checking if night splitting works correctly
        assert all(isinstance(d.index, pd.DatetimeIndex) for d in data.values())
        # only 1 night
        assert len(data) == 1
        TestCase().assertListEqual(list(data.keys()), ["2020-10-23"])
        # data has a duration of 16 minutes (after interpolating to 1 min equidistant index)
        assert all(len(d.index) == 16 for d in data.values())
        assert all(str(d.index.tz) == "Europe/Berlin" for d in data.values())

    @pytest.mark.parametrize(
        "file_path, data_source",
        [
            ("raw_sleep-monitor_hr.csv", "heart_rate"),
        ],
    )
    def test_load_withings_sleep_analyzer_raw_file_no_split(self, file_path, data_source):
        data = load_withings_sleep_analyzer_raw_file(
            TEST_FILE_PATH.joinpath("sleep_analyzer").joinpath(file_path),
            data_source=data_source,
            split_into_nights=False,
        )
        assert isinstance(data.index, pd.DatetimeIndex)
        # data has a duration of 16 minutes
        assert len(data.index) == 16
        assert str(data.index.tz) == "Europe/Berlin"

    @pytest.mark.parametrize(
        "folder_path, expected",
        [
            ("sleep_analyzer", does_not_raise()),
            ("sleep_analyzer_empty", pytest.raises(ValueError)),
            ("sleep_analyzer_summary.csv", pytest.raises(ValueError)),
        ],
    )
    def test_load_withings_sleep_analyzer_raw_folder_raises(self, folder_path, expected):
        with expected:
            load_withings_sleep_analyzer_raw_folder(TEST_FILE_PATH.joinpath(folder_path))

    @pytest.mark.parametrize(
        "folder_path",
        [
            ("sleep_analyzer"),
        ],
    )
    def test_load_withings_sleep_analyzer_raw_folder(self, folder_path):
        data = load_withings_sleep_analyzer_raw_folder(TEST_FILE_PATH.joinpath(folder_path))
        # data has a duration of 16 minutes
        assert isinstance(data, dict)
        assert len(data) == 1
        assert len(list(data.values())[0].index) == 16
        TestCase().assertListEqual(
            list(list(data.values())[0].columns), ["heart_rate", "respiration_rate", "sleep_state", "snoring"]
        )

    @pytest.mark.parametrize(
        "folder_path",
        [
            ("sleep_analyzer"),
        ],
    )
    def test_load_withings_sleep_analyzer_raw_folder_no_split(self, folder_path):
        data = load_withings_sleep_analyzer_raw_folder(TEST_FILE_PATH.joinpath(folder_path), split_into_nights=False)
        assert isinstance(data, pd.DataFrame)
        # data has a duration of 16 minutes (when split_into_nights is False)
        assert len(data.index) == 16
        TestCase().assertListEqual(list(data.columns), ["heart_rate", "respiration_rate", "sleep_state", "snoring"])
