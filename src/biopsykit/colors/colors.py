"""Color palettes and utilities.

For most plots, BioPsyKit uses FAU's colors scheme.
"""
from typing import Sequence, Optional, Dict, Callable

from colorsys import rgb_to_hls, hls_to_rgb
from typing_extensions import Literal, get_args

from matplotlib.colors import to_hex, to_rgb
import seaborn as sns

__all__ = [
    "FAU_COLORS",
    "fau_palette_by_name",
    "fau_color_dict",
    "fau_palette",
    "fau_palette_blue",
    "fau_palette_wiso",
    "fau_palette_phil",
    "fau_palette_tech",
    "fau_palette_nat",
    "fau_palette_med",
    "fau_color",
    "adjust_color",
]

FAU_COLORS = Literal["fau", "tech", "phil", "med", "nat", "wiso"]
"""
Available color keys.
"""

fau_color_dict: Dict[str, str] = {
    "fau": "#003865",
    "tech": "#98a4ae",
    "phil": "#c99313",
    "med": "#00b1eb",
    "nat": "#009b77",
    "wiso": "#8d1429",
}
"""
Dictionary for FAU color codes.
"""

fau_palette = sns.color_palette(fau_color_dict.values())  #: :meta hide-value:
"""FAU color palette that can be used with seaborn and matplotlib."""


def fau_palette_by_name(name: FAU_COLORS) -> Callable:
    """Return the function to create a FAU color palette by the name the color.

    Parameters
    ----------
    name : str
        Color name. Must be one of :const:`biopsykit.colors.FAU_COLORS`.

    Returns
    -------
    function
        function to create FAU color palette with

    Examples
    --------
    >>> from biopsykit.colors import fau_palette_by_name
    >>> fau_palette_by_name("tech")
    <function biopsykit.colors.colors.fau_palette_tech()>

    """
    if name not in get_args(FAU_COLORS):
        raise ValueError("'name' must be one of {}, got '{}'!".format(get_args(FAU_COLORS), name))
    if name == "fau":
        name = "blue"
    return globals()["fau_palette_{}".format(name)]


def fau_palette_blue(n_colors: Optional[int] = 8) -> Sequence[str]:
    """Return a seaborn palette with fau-blue color nuances.

    Parameters
    ----------
    n_colors : int
        number of colors in the palette. Default: 8

    Returns
    -------
    list of tuple
        list of RGB tuples

    """
    palette = sns.light_palette("#003865", n_colors=10, reverse=True)[:-2]
    if n_colors <= 4:
        palette = palette[:: len(palette) - (n_colors + 2)]
    else:
        palette = palette[:n_colors]

    return palette


def fau_palette_tech(n_colors: Optional[int] = 8) -> Sequence[str]:
    """Return a seaborn palette with fau-tech-grey color nuances.

    Parameters
    ----------
    n_colors : int
        number of colors in the palette. Default: 8

    Returns
    -------
    list of tuple
        list of RGB tuples

    """
    palette = sns.dark_palette("#98a4ae", n_colors=10, reverse=True)[:-2]
    if n_colors <= 4:
        palette = palette[:: len(palette) - (n_colors + 2)]
    else:
        palette = palette[:n_colors]

    return palette


def fau_palette_phil(n_colors: Optional[int] = 8) -> Sequence[str]:
    """Return a seaborn palette with fau-phil-yellow color nuances.

    Parameters
    ----------
    n_colors : int
        number of colors in the palette. Default: 8

    Returns
    -------
    list of tuple
        list of RGB tuples

    """
    palette = sns.dark_palette("#c99313", n_colors=10, reverse=True)[:-2]
    if n_colors <= 4:
        palette = palette[:: len(palette) - (n_colors + 2)]
    else:
        palette = palette[:n_colors]

    return palette


def fau_palette_med(n_colors: Optional[int] = 8) -> Sequence[str]:
    """Return a seaborn palette with fau-med-light-blue color nuances.

    Parameters
    ----------
    n_colors : int
        number of colors in the palette. Default: 8

    Returns
    -------
    list of tuple
        list of RGB tuples

    """
    palette = sns.dark_palette("#00b1eb", n_colors=10, reverse=True)[:-2]
    if n_colors <= 4:
        palette = palette[:: len(palette) - (n_colors + 2)]
    else:
        palette = palette[:n_colors]

    return palette


def fau_palette_nat(n_colors: Optional[int] = 8) -> Sequence[str]:
    """Return a seaborn palette with fau-nat-green color nuances.

    Parameters
    ----------
    n_colors : int
        number of colors in the palette. Default: 8

    Returns
    -------
    list of tuple
        list of RGB tuples

    """
    palette = sns.dark_palette("#009b77", n_colors=10, reverse=True)[:-2]
    if n_colors <= 4:
        palette = palette[:: len(palette) - (n_colors + 2)]
    else:
        palette = palette[:n_colors]

    return palette


def fau_palette_wiso(n_colors: Optional[int] = 8) -> Sequence[str]:
    """Return a seaborn palette with fau-wiso-red color nuances.

    Parameters
    ----------
    n_colors : int
        number of colors in the palette. Default: 8

    Returns
    -------
    list of tuple
        list of RGB tuples

    """
    palette = sns.light_palette("#8d1429", n_colors=10, reverse=True)[:-2]
    if n_colors <= 4:
        palette = palette[:: len(palette) - (n_colors + 2)]
    else:
        palette = palette[:n_colors]

    return palette


def fau_color(color: FAU_COLORS) -> str:
    """Return the color specified by ``color`` as hex string.

    Parameters
    ----------
    color : str
        Color key. Must be one of :const:`biopsykit.colors.FAU_COLORS`.

    Returns
    -------
    str
        color as hex string

    """
    if color not in get_args(FAU_COLORS):
        raise ValueError("'color' must be one of {}, got '{}'!".format(get_args(FAU_COLORS), color))
    return fau_color_dict[color]


def adjust_color(key: FAU_COLORS, amount: Optional[float] = 1.5) -> str:
    """Adjust a FAU color in its brightness.

    Parameters
    ----------
    key : str
        Color key. Must be one of :const:`biopsykit.colors.FAU_COLORS`.
    amount : float, optional
        Parameter to adjust brightness. An ``amount`` value < 1 results in a darker color,
        an ``amount`` value > 1 results in a brighter color.

    Returns
    -------
    str
        adjusted FAU color as hex code

    """
    c = rgb_to_hls(*to_rgb(fau_color(key)))
    return to_hex(hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2]))
