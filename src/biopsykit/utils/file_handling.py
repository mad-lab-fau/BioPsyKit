"""Helper functions for file handling."""
from pathlib import Path
from typing import Sequence, Optional, Union

from biopsykit.utils._datatype_validation_helper import _assert_file_extension
from biopsykit.utils._types import path_t


def mkdirs(dir_list: Union[path_t, Sequence[path_t]]) -> None:
    """Batch-create a list of directories.

    Conveniently create a list of directories, e.g. directories for storing processing
    results, figures, and statistic reports, using :any:`~pathlib.Path.mkdir`.
    If parent directories do not exist yet, they are created (`pathlib` option ``parent=True``),
    if directories already exist no Error is thrown (`pathlib` option ``exist_ok=True``).

    Parameters
    ----------
    dir_list : list of path or str
        list of directory names to create

    Examples
    --------
    >>> from biopsykit.utils.file_handling import mkdirs
    >>> path_list = [Path("processing_results"), Path("exports/figures"), Path("exports/statistics")]
    >>> mkdirs(path_list)

    """
    if isinstance(dir_list, (str, Path)):
        dir_list = [dir_list]
    for directory in dir_list:
        # ensure pathlib
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)


def get_subject_dirs(base_path: path_t, pattern: str) -> Sequence[Path]:
    """Filter for subject directories using a name pattern.

    Parameters
    ----------
    base_path : path or str
        base path to filter for directories
    pattern : str
        name pattern as regex

    Returns
    -------
    list of path
        a list of path or an empty list if no subfolders matched the``pattern``

    Examples
    --------
    >>> from biopsykit.utils.file_handling import get_subject_dirs
    >>> base_path = Path(".")
    >>> get_subject_dirs(base_path, "Vp*")

    """
    # ensure pathlib
    base_path = Path(base_path)
    return [p for p in sorted(base_path.glob(pattern)) if p.is_dir()]


def is_excel_file(file_name: path_t, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether the file name is an Excel file.

    Parameters
    ----------
    file_name : :any:`pathlib.Path` or str
        file name to check
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``file_name`` is an Excel file, i.e. has the suffix ``.xlsx``, ``False`` otherwise
    (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``file_name`` is not an Excel file

    """
    return _assert_file_extension(file_name, (".xlsx", ".xls"), raise_exception)
