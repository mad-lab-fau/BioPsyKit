"""Helper functions for file handling."""
import re
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt

from biopsykit.utils._datatype_validation_helper import _assert_file_extension
from biopsykit.utils._types import path_t


def mkdirs(dir_list: Union[path_t, Sequence[path_t]]) -> None:
    """Batch-create a list of directories.

    Conveniently create a list of directories, e.g. directories for storing processing
    results, figures, and statistic reports, using :meth:`~pathlib.Path.mkdir`.
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


def get_subject_dirs(base_path: path_t, pattern: str) -> Optional[Sequence[Path]]:
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
        a list of path or an empty list if no subfolders matched the ``pattern``

    Raises
    ------
    FileNotFoundError
        if no subfolders in ``base_path`` match ``pattern``.

    Examples
    --------
    >>> from biopsykit.utils.file_handling import get_subject_dirs
    >>> base_path = Path(".")
    >>> get_subject_dirs(base_path, "Vp*")

    """
    # ensure pathlib
    base_path = Path(base_path)
    subject_dirs = [p for p in sorted(base_path.glob("*")) if p.is_dir()]
    subject_dirs = list(filter(lambda s: len(re.findall(pattern, s.name)) > 0, subject_dirs))
    if len(subject_dirs) == 0:
        raise FileNotFoundError("No subfolders matching the pattern '{}' found in {}.".format(pattern, base_path))
    return subject_dirs


def export_figure(
    fig: plt.Figure,
    filename: path_t,
    base_dir: path_t,
    formats: Optional[Sequence[str]] = None,
    use_subfolder: Optional[bool] = True,
):
    """Export matplotlib figure to file(s).

    This function allows to export a matplotlib figure in multiple file formats, each format being optionally saved
    in its own output subfolder.

    Parameters
    ----------
    fig : :class:`~matplotlib.figure.Figure`
        matplotlib figure object
    filename : :class:`~pathlib.Path` or str
        name of the output file
    base_dir : path or str
        base directory to export figures to
    formats: list of str, optional
        list of file formats to export or ``None`` to export as pdf. Default: ``None``
    use_subfolder : bool, optional
        whether to create an own output subfolder per file format or not. Default: ``True``

    Examples
    --------
    >>> fig = plt.Figure()
    >>>
    >>> base_dir = "./img"
    >>> filename = "plot"
    >>> formats = ["pdf", "png"]
    >>> # Export into subfolders (default)
    >>> export_figure(fig, filename=filename, base_dir=base_dir, formats=formats)
    >>> # | img/
    >>> # | - pdf/
    >>> # | - - plot.pdf
    >>> # | - png/
    >>> # | - - plot.png

    >>> # Export into one folder
    >>> export_figure(fig, filename=filename, base_dir=base_dir, formats=formats, use_subfolder=False)
    >>> # | img/
    >>> # | - plot.pdf
    >>> # | - plot.png

    """
    if formats is None:
        formats = ["pdf"]
    # ensure list
    if isinstance(formats, str):
        formats = [formats]

    # ensure pathlib
    base_dir = Path(base_dir)
    filename = Path(filename)
    subfolders = [base_dir] * len(formats)

    if use_subfolder:
        subfolders = [base_dir.joinpath(f) for f in formats]
        for folder in subfolders:
            folder.mkdir(exist_ok=True, parents=True)

    for f, subfolder in zip(formats, subfolders):
        fig.savefig(
            subfolder.joinpath(filename.name + "." + f),
            transparent=(f == "pdf"),
            format=f,
        )


def is_excel_file(file_name: path_t, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether the file name is an Excel file.

    Parameters
    ----------
    file_name : :class:`~pathlib.Path` or str
        file name to check
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``file_name`` is an Excel file, i.e. has the suffix ``.xlsx``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``file_name`` is not an Excel file

    """
    return _assert_file_extension(file_name, (".xlsx", ".xls"), raise_exception)
