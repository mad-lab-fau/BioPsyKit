# -*- coding: utf-8 -*-
"""Set of helper functions used throughout the library.

@author: Robert Richer, Arne Küderle
"""
from pathlib import Path
from typing import Sequence, Optional

from biopsykit._types import path_t


def export_figure(fig: 'plt.Figure', filename: path_t, base_dir: path_t, formats: Optional[Sequence[str]] = None,
                  use_subfolder: Optional[bool] = True) -> None:
    """
    Exports a figure to a file.

    Parameters
    ----------
    fig : Figure
        matplotlib figure object
    filename : path or str
        name of the output file
    base_dir : path or str
        base directory to save file
    formats: list of str, optional
        list of file formats to export or ``None`` to export as pdf. Default: ``None``
    use_subfolder : bool, optional
        whether to create an own subfolder per file format and export figures into these subfolders. Default: True

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import biopsykit as bp
    >>> fig = plt.Figure()
    >>>
    >>> base_dir = "./img"
    >>> filename = "plot"
    >>> formats = ["pdf", "png"]

    >>> # Export into subfolders (default)
    >>> bp.plotting.export_figure(fig, filename=filename, base_dir=base_dir, formats=formats)
    >>> # | img/
    >>> # | - pdf/
    >>> # | - - plot.pdf
    >>> # | - png/
    >>> # | - - plot.png

    >>> # Export into one folder
    >>> bp.plotting.export_figure(fig, filename=filename, base_dir=base_dir, formats=formats, use_subfolder=False)
    >>> # | img/
    >>> # | - plot.pdf
    >>> # | - plot.png
    """
    import matplotlib.pyplot as plt

    if formats is None:
        formats = ['pdf']

    # ensure pathlib
    base_dir = Path(base_dir)
    filename = Path(filename)
    subfolders = [base_dir] * len(formats)

    if use_subfolder:
        subfolders = [base_dir.joinpath(f) for f in formats]
        for folder in subfolders:
            folder.mkdir(exist_ok=True, parents=True)

    for f, subfolder in zip(formats, subfolders):
        fig.savefig(subfolder.joinpath(filename.name + '.' + f), transparent=(f == 'pdf'), format=f)


