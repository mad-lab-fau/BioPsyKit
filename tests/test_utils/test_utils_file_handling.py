from contextlib import contextmanager
from pathlib import Path

import pytest

from biopsykit.utils.exceptions import FileExtensionError
from biopsykit.utils.file_handling import get_subject_dirs, is_excel_file, mkdirs

TEST_FILE_PATH = Path(__file__).parent.joinpath("../test_data/load_hr_subject_dict_folder")


@contextmanager
def does_not_raise():
    yield


class TestUtilsFileHandling:
    @pytest.mark.parametrize("dir_list", [["test1", "test2", "test3"], "test_dir"])
    def test_mkdirs(self, dir_list, tmp_path):
        # ensure pathlib
        tmp_path = Path(tmp_path)
        if isinstance(dir_list, str):
            # only one directory as string
            dir_list = tmp_path.joinpath(dir_list)
        else:
            # list of directories
            dir_list = [tmp_path.joinpath(s) for s in dir_list]
        mkdirs(dir_list)
        ac_dir_list = [p for p in tmp_path.glob("*") if p.is_dir()]
        if isinstance(dir_list, Path):
            # only one directory as string (now it's a path)
            assert dir_list in ac_dir_list
        else:
            # list of directories
            assert all([p in ac_dir_list for p in dir_list])

    @pytest.mark.parametrize(
        "base_path, pattern, expected",
        [
            (TEST_FILE_PATH, "Vp*", does_not_raise()),
            (TEST_FILE_PATH.joinpath("dir"), "Vp*", pytest.raises(FileNotFoundError)),
        ],
    )
    def test_get_subject_dirs_raises(self, base_path, pattern, expected):
        with expected:
            get_subject_dirs(base_path, pattern)

    @pytest.mark.parametrize("base_path, pattern, expected", [(TEST_FILE_PATH, "Vp*", ["Vp03", "Vp04", "Vp05"])])
    def test_get_subject_dirs(self, base_path, pattern, expected):
        subject_dirs = get_subject_dirs(base_path, pattern)
        assert all([TEST_FILE_PATH.joinpath(p) in subject_dirs for p in expected])

    @pytest.mark.parametrize(
        "file_name, expected",
        [
            ("test.xlsx", does_not_raise()),
            ("test.xls", does_not_raise()),
            ("test.csv", pytest.raises(FileExtensionError)),
            ("", pytest.raises(FileExtensionError)),
        ],
    )
    def test_is_excel_file_exception(self, file_name, expected):
        with expected:
            is_excel_file(file_name)

    @pytest.mark.parametrize(
        "file_name, expected",
        [
            ("test.xlsx", True),
            ("test.xls", True),
            ("test.csv", False),
            ("", False),
        ],
    )
    def test_is_excel_file_bool(self, file_name, expected):
        assert is_excel_file(file_name, raise_exception=False) == expected
