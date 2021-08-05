"""Color palettes and utilities.

For most plots, BioPsyKit uses FAU's colors scheme.
"""
from typing import Union, Sequence, Optional, Dict

from colorsys import rgb_to_hls, hls_to_rgb
from matplotlib.colors import to_hex, to_rgb
import seaborn as sns

__all__ = [
    "FAU_COLORS",
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

FAU_COLORS = ["fau", "tech", "phil", "med", "nat", "wiso"]
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


def fau_palette_blue(palette_type: Optional[str] = None) -> Sequence[str]:
    """Return a seaborn palette with fau-blue color nuances.

    By default, a palette with 10 color nuances is generated.
    Using the ``palette_type`` parameter specialized palettes can be generated.

    Parameters
    ----------
    palette_type : str, optional
        Specify specialized color palette (or ``None`` to return default palette). Default: ``None``.
        Available palette types:

        * ``line_2``: For line plots with two elements
        * ``line_3``: For line plots with three elements
        * ``ensemble_3``: For ensemble plots (mean ± std) with three elements
        * ``box_2``: For boxplots with two elements

    Returns
    -------
    list of tuple
        list of RGB tuples

    """
    # generated using this link: https://noeldelgado.github.io/shadowlord
    fau_blue = sns.color_palette(
        [
            "#001628",
            "#001c33",
            "#002747",
            "#002d51",
            "#003865",
            "#194c74",
            "#336084",
            "#4d7493",
            "#809cb2",
            "#b3c3d1",
            "#e6ebf0",
        ]
    )
    if palette_type == "line_3":
        return [fau_blue[4], fau_blue[7], fau_blue[9]]
    if palette_type == "ensemble":
        return fau_blue[1::2]
    if palette_type == "box_2":
        return fau_blue[5::4]
    if palette_type == "line_2":
        return fau_blue[2::5]
    return fau_blue


def fau_palette_wiso(palette_type: Optional[str] = None) -> Sequence[str]:
    """Return a seaborn palette with fau-wiso-red color nuances.

    By default, a palette with 10 color nuances is generated.
    Using the ``palette_type`` parameter specialized palettes can be generated.

    Parameters
    ----------
    palette_type : str, optional
        Specify specialized color palette (or ``None`` to return default palette). Default: ``None``.
        Available palette types:

        * ``line_2``: For line plots with two elements
        * ``line_3``: For line plots with three elements
        * ``ensemble_3``: For ensemble plots (mean ± std) with three elements
        * ``box_2``: For boxplots with two elements

    Returns
    -------
    list of tuple
        list of RGB tuples

    """
    # generated using this link: https://noeldelgado.github.io/shadowlord
    fau_wiso = sns.color_palette(
        [
            "#1c0408",
            "#2a060c",
            "#380810",
            "#470a15",
            "#550c19",
            "#711021",
            "#8d1429",
            "#a44354",
            "#bb727f",
            "#d1a1a9",
            "#e8d0d4",
        ]
    )

    if palette_type == "line_3":
        return [fau_wiso[4], fau_wiso[7], fau_wiso[9]]
    if palette_type == "ensemble_3":
        return [fau_wiso[3], fau_wiso[5], fau_wiso[8]]
    if palette_type == "box_2":
        return fau_wiso[5::4]
    if palette_type == "line_2":
        return fau_wiso[2::5]
    return fau_wiso


def fau_palette_phil(palette_type: Union[str, None]) -> Sequence[str]:
    """Return a seaborn palette with fau-phil-yellow color nuances.

    By default, a palette with 10 color nuances is generated.
    Using the ``palette_type`` parameter specialized palettes can be generated.

    Parameters
    ----------
    palette_type : str, optional
        Specify specialized color palette (or ``None`` to return default palette). Default: ``None``.
        Available palette types:

        * ``line_2``: For line plots with two elements
        * ``line_3``: For line plots with three elements
        * ``ensemble_3``: For ensemble plots (mean ± std) with three elements
        * ``box_2``: For boxplots with two elements

    Returns
    -------
    list of tuple
        list of RGB tuples

    """
    # generated using this link: https://noeldelgado.github.io/shadowlord
    fau_phil = sns.color_palette(
        [
            "#3c2c06",
            "#503b08",
            "#654a0a",
            "#79580b",
            "#a1760f",
            "#c99313",
            "#d4a942",
            "#dfbe71",
            "#e4c989",
            "#e9d4a1",
            "#f4e9d0",
        ]
    )
    if palette_type == "line_3":
        return [fau_phil[4], fau_phil[7], fau_phil[9]]
    if palette_type == "ensemble_3":
        return [fau_phil[3], fau_phil[5], fau_phil[8]]
    if palette_type == "box_2":
        return fau_phil[5::4]
    if palette_type == "line_2":
        return fau_phil[2::5]
    return fau_phil


def fau_palette_med(palette_type: Union[str, None]) -> Sequence[str]:
    """Return a seaborn palette with fau-med-light-blue color nuances.

    By default, a palette with 10 color nuances is generated.
    Using the ``palette_type`` parameter specialized palettes can be generated.

    Parameters
    ----------
    palette_type : str, optional
        Specify specialized color palette (or ``None`` to return default palette). Default: ``None``.
        Available palette types:

        * ``line_2``: For line plots with two elements
        * ``line_3``: For line plots with three elements
        * ``ensemble_3``: For ensemble plots (mean ± std) with three elements
        * ``box_2``: For boxplots with two elements

    Returns
    -------
    list of tuple
        list of RGB tuples

    """
    # generated using this link: https://noeldelgado.github.io/shadowlord
    fau_med = sns.color_palette(
        [
            "#00232f",
            "#003547",
            "#00475e",
            "#005976",
            "#006a8d",
            "#008ebc",
            "#00b1eb",
            "#33c1ef",
            "#66d0f3",
            "#99e0f7",
            "#cceffb",
        ]
    )
    if palette_type == "line_3":
        return [fau_med[4], fau_med[7], fau_med[9]]
    if palette_type == "ensemble_3":
        return [fau_med[3], fau_med[5], fau_med[8]]
    if palette_type == "box_2":
        return fau_med[5::4]
    if palette_type == "line_2":
        return fau_med[2::5]
    return fau_med


def fau_palette_nat(palette_type: Union[str, None]) -> Sequence[str]:
    """Return a seaborn palette with fau-med-light-blue color nuances.

    By default, a palette with 10 color nuances is generated.
    Using the ``palette_type`` parameter specialized palettes can be generated.

    Parameters
    ----------
    palette_type : str, optional
        Specify specialized color palette (or ``None`` to return default palette). Default: ``None``.
        Available palette types:

        * ``line_2``: For line plots with two elements
        * ``line_3``: For line plots with three elements
        * ``ensemble_3``: For ensemble plots (mean ± std) with three elements
        * ``box_2``: For boxplots with two elements

    Returns
    -------
    list of tuple
        list of RGB tuples

    """
    # generated using this link: https://noeldelgado.github.io/shadowlord
    fau_nat = sns.color_palette(
        [
            "#001f18",
            "#002f24",
            "#003e30",
            "#004e3c",
            "#005d47",
            "#007c5f",
            "#009b77",
            "#33af92",
            "#66c3ad",
            "#99d7c9",
            "#ccebe4",
        ]
    )
    if palette_type == "line_3":
        return [fau_nat[4], fau_nat[7], fau_nat[9]]
    if palette_type == "ensemble_3":
        return [fau_nat[3], fau_nat[5], fau_nat[8]]
    if palette_type == "box_2":
        return fau_nat[5::4]
    if palette_type == "line_2":
        return fau_nat[2::5]
    return fau_nat


def fau_palette_tech(palette_type: Union[str, None]) -> Sequence[str]:
    """Return a seaborn palette with fau-tech-grey color nuances.

    By default, a palette with 10 color nuances is generated.
    Using the ``palette_type`` parameter specialized palettes can be generated.

    Parameters
    ----------
    palette_type : str, optional
        Specify specialized color palette (or ``None`` to return default palette). Default: ``None``.
        Available palette types:

        * ``line_2``: For line plots with two elements
        * ``line_3``: For line plots with three elements
        * ``ensemble_3``: For ensemble plots (mean ± std) with three elements
        * ``box_2``: For boxplots with two elements

    Returns
    -------
    list of tuple
        list of RGB tuples

    """
    # generated using this link: https://noeldelgado.github.io/shadowlord
    fau_tech = sns.color_palette(
        [
            "#1e2123",
            "#2e3134",
            "#3d4246",
            "#5b6268",
            "#6a737a",
            "#7a838b",
            "#98a4ae",
            "#adb6be",
            "#b7bfc6",
            "#c1c8ce",
            "#d6dbdf",
        ]
    )
    if palette_type == "line_3":
        return [fau_tech[4], fau_tech[7], fau_tech[9]]
    if palette_type == "ensemble_3":
        return [fau_tech[3], fau_tech[5], fau_tech[8]]
    if palette_type == "box_2":
        return fau_tech[4::4]
    if palette_type == "line_2":
        return fau_tech[3::4][::-1]
    return fau_tech


def fau_color(color: str) -> str:
    """Return the color specified by ``color`` as hex string.

    Parameters
    ----------
    color : str
        Color key. Must be one of :const:`biopsykit.colors.FAU_COLORS`

    Returns
    -------
    str
        color as hex string

    """
    return fau_color_dict[color]


def adjust_color(key: str, amount: Optional[float] = 1.5) -> str:
    """Adjust a FAU color in its brightness.

    Parameters
    ----------
    key : str
        color string
    amount : float, optional
        Parameter to adjust brightness. An ``amount`` value < 1 results in a darker color,
        an ``amount`` value > 1 results in a brighter color.

    Returns
    -------
    str
        adjusted color as hex code

    """
    c = rgb_to_hls(*to_rgb(fau_color(key)))
    return to_hex(hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2]))
