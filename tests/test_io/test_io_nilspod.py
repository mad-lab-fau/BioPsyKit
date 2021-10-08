from contextlib import contextmanager
from pathlib import Path
from unittest import TestCase

import pandas as pd
import pytest
from nilspodlib import Dataset

from biopsykit.io.nilspod import (
    check_nilspod_dataset_corrupted,
    get_nilspod_dataset_corrupted_info,
    load_csv_nilspod,
    load_dataset_nilspod,
    load_folder_nilspod,
    load_synced_session_nilspod,
)

TEST_FILE_PATH = Path(__file__).parent.joinpath("../test_data/nilspod")


@contextmanager
def does_not_raise():
    yield


class TestIoNilspod:
    @pytest.mark.parametrize(
        "file_path, dataset, expected",
        [
            (TEST_FILE_PATH.joinpath("test_dataset.bin"), None, does_not_raise()),
            (
                None,
                Dataset.from_bin_file(TEST_FILE_PATH.joinpath("test_dataset.bin"), tz="Europe/Berlin"),
                does_not_raise(),
            ),
            (
                None,
                Dataset.from_bin_file(TEST_FILE_PATH.joinpath("test_dataset.bin")),
                pytest.raises(ValueError),
            ),
            (None, None, pytest.raises(ValueError)),
        ],
    )
    def test_load_dataset_nilspod_raises(
        self,
        file_path,
        dataset,
        expected,
    ):
        with expected:
            load_dataset_nilspod(file_path=file_path, dataset=dataset)

    @pytest.mark.parametrize(
        "file_path, dataset, datastreams, timezone",
        [
            ("test_dataset.bin", None, None, None),
            ("test_dataset.bin", None, ["acc", "gyro"], "Europe/Berlin"),
            ("test_dataset.bin", None, "acc", "Europe/London"),
        ],
    )
    def test_load_dataset_nilspod(self, file_path, dataset, datastreams, timezone):
        df, fs = load_dataset_nilspod(
            file_path=TEST_FILE_PATH.joinpath(file_path), dataset=dataset, datastreams=datastreams
        )
        assert isinstance(df, pd.DataFrame)
        assert type(fs) == float

    @pytest.mark.parametrize(
        "folder_path, expected",
        [
            ("synced_sample_session", does_not_raise()),
            ("synced_sample_session_empty", pytest.raises(ValueError)),
        ],
    )
    def test_load_synced_session_nilspod_raises(self, folder_path, expected):
        with expected:
            load_synced_session_nilspod(folder_path=TEST_FILE_PATH.joinpath(folder_path), legacy_support="warn")

    @pytest.mark.parametrize(
        "folder_path, datastreams, timezone",
        [
            ("synced_sample_session", None, None),
            ("synced_sample_session", "acc", None),
            ("synced_sample_session", "acc", "Europe/Berlin"),
        ],
    )
    def test_load_synced_session_nilspod(self, folder_path, datastreams, timezone):
        df, fs = load_synced_session_nilspod(
            folder_path=TEST_FILE_PATH.joinpath(folder_path), timezone=timezone, legacy_support="warn"
        )
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.columns, pd.MultiIndex)
        assert type(fs) == float

    @pytest.mark.parametrize(
        "folder_path, phase_names, expected",
        [
            ("multi_recordings", None, does_not_raise()),
            ("multi_recordings", ["Start", "End"], does_not_raise()),
            ("multi_recordings", ["Start", "Middle", "End"], pytest.raises(ValueError)),
            ("multi_recordings_sampling_rate", None, pytest.raises(ValueError)),
            ("multi_recordings_empty", None, pytest.raises(ValueError)),
        ],
    )
    def test_load_folder_nilspod_raises(self, folder_path, phase_names, expected):
        with expected:
            load_folder_nilspod(
                folder_path=TEST_FILE_PATH.joinpath(folder_path), phase_names=phase_names, legacy_support="warn"
            )

    @pytest.mark.parametrize(
        "file_path, expected",
        [("test_dataset.bin", False), ("test_dataset_corrupted.bin", True)],
    )
    def test_check_nilspod_dataset_corrupted(self, file_path, expected):
        assert check_nilspod_dataset_corrupted(Dataset.from_bin_file(TEST_FILE_PATH.joinpath(file_path))) == expected

    @pytest.mark.parametrize(
        "file_path, expected",
        [
            ("test_dataset.bin", {"name": "test_dataset.bin", "percent_corrupt": 0.0, "condition": "fine"}),
            (
                "test_dataset_corrupted.bin",
                {"name": "test_dataset_corrupted.bin", "percent_corrupt": 2.8, "condition": "end_only"},
            ),
        ],
    )
    def test_get_nilspod_dataset_corrupted_info(self, file_path, expected):
        data_out = get_nilspod_dataset_corrupted_info(
            Dataset.from_bin_file(TEST_FILE_PATH.joinpath(file_path)), file_path
        )
        TestCase().assertDictEqual(data_out, expected)

    @pytest.mark.parametrize(
        "file_path, filename_regex, time_regex, expected",
        [
            ("NilsPodX-7FAD_20190430_093300.csv", None, None, does_not_raise()),
            ("NilsPodX-7FAD_20190430_093300.csv", None, None, does_not_raise()),
            ("test_dataset.csv", None, None, does_not_raise()),
            ("test_dataset_20190430_093300.csv", "test_dataset_(.*?).csv", None, does_not_raise()),
            ("test_dataset_20190430_093300.csv", "test_dataset_(.*?).csv", "test", pytest.raises(ValueError)),
        ],
    )
    def test_load_csv_nilspod_raises(self, file_path, filename_regex, time_regex, expected):
        with expected:
            load_csv_nilspod(TEST_FILE_PATH.joinpath(file_path), filename_regex=filename_regex, time_regex=time_regex)

    @pytest.mark.parametrize(
        "file_path, filename_regex, time_regex, datastreams, expected_index_type, expected_columns",
        [
            (
                "NilsPodX-7FAD_20190430_093300.csv",
                None,
                None,
                None,
                pd.DatetimeIndex,
                ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "ecg"],
            ),
            (
                "NilsPodX-7FAD_20190430_093300.csv",
                None,
                None,
                ["acc", "ecg"],
                pd.DatetimeIndex,
                ["acc_x", "acc_y", "acc_z", "ecg"],
            ),
            (
                "test_dataset.csv",
                None,
                None,
                "ecg",
                pd.TimedeltaIndex,
                ["ecg"],
            ),
            (
                "test_dataset_20190430_093300.csv",
                r"test_dataset_(\w+).csv",
                None,
                "ecg",
                pd.DatetimeIndex,
                ["ecg"],
            ),
        ],
    )
    def test_load_csv_nilspod(
        self, file_path, filename_regex, time_regex, datastreams, expected_index_type, expected_columns
    ):
        df, fs = load_csv_nilspod(
            TEST_FILE_PATH.joinpath(file_path),
            filename_regex=filename_regex,
            time_regex=time_regex,
            datastreams=datastreams,
        )
        assert fs == 256.0
        assert isinstance(df.index, expected_index_type)
        TestCase().assertListEqual(list(df.columns), list(expected_columns))

    @pytest.mark.parametrize(
        "file_path, time_regex, expected_time",
        [
            ("NilsPodX-7FAD_20190430_0933.csv", None, "2019-04-30 09:03:03+02:00"),
            ("NilsPodX-7FAD_20190430_0933.csv", "%Y%m%d_%H%M", "2019-04-30 09:33:00+02:00"),
        ],
    )
    def test_load_csv_nilspod_time(self, file_path, time_regex, expected_time):
        df, fs = load_csv_nilspod(
            TEST_FILE_PATH.joinpath(file_path),
            time_regex=time_regex,
        )
        assert str(df.index[0]) == expected_time
