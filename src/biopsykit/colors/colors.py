"""
Color palettes and utilities.

For most plots, BioPsyKit uses FAU's colors scheme.
"""
from typing import Union, Sequence, Optional, Literal, TypedDict

import seaborn as sns

__all__ = ["fau_color_dict", "fau_palette"]

FAU_COLORS = Literal["fau", "tech", "phil", "med", "nat", "wiso"]

fau_color_dict: TypedDict[FAU_COLORS, str] = {
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

fau_palette = sns.color_palette(fau_color_dict.values())
"""FAU color palette that can be used with seaborn and matplotlib."""


def cmap_fau_blue(cmap_type: Optional[str] = None) -> Sequence[str]:
    """


    Parameters
    ----------
    cmap_type : str, optional

    Returns
    -------

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
    if cmap_type == "3":
        return [fau_blue[4], fau_blue[7], fau_blue[9]]
    if cmap_type == "3_ens":
        return [fau_blue[3], fau_blue[5], fau_blue[8]]
    elif cmap_type == "2":
        return fau_blue[5::4]
    elif cmap_type == "2_lp":
        return fau_blue[2::5]
    else:
        return fau_blue


def cmap_fau_wiso(cmap_type: Union[str, None]) -> Sequence[str]:
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
    if cmap_type == "3":
        return fau_wiso[1::3]
    elif cmap_type == "2":
        return fau_wiso[5::4]
    elif cmap_type == "2_lp":
        return fau_wiso[2::5]
    else:
        return fau_wiso


def cmap_fau_phil(cmap_type: Union[str, None]) -> Sequence[str]:
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
    if cmap_type == "3":
        return fau_phil[1::3]
    elif cmap_type == "2":
        return fau_phil[5::4]
    elif cmap_type == "2_lp":
        return fau_phil[2::5]
    else:
        return fau_phil


def cmap_fau_med(cmap_type: Union[str, None]) -> Sequence[str]:
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
    if cmap_type == "3":
        return fau_med[1::3]
    elif cmap_type == "2":
        return fau_med[5::4]
    elif cmap_type == "2_lp":
        return fau_med[2::5]
    else:
        return fau_med


def cmap_fau_nat(cmap_type: Union[str, None]) -> Sequence[str]:
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
    if cmap_type == "3":
        return fau_nat[1::3]
    elif cmap_type == "2":
        return fau_nat[5::4]
    elif cmap_type == "2_lp":
        return fau_nat[2::5]
    else:
        return fau_nat


def cmap_fau_tech(cmap_type: Union[str, None]) -> Sequence[str]:
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
    if cmap_type == "3":
        return fau_tech[1::3]
    elif cmap_type == "2":
        return fau_tech[5::4]
    elif cmap_type == "2_lp":
        return fau_tech[2::5]
    else:
        return fau_tech


def fau_color(color: str) -> str:
    """
    Returns the FAU color specified by `color`. Must be one of
    Parameters
    ----------
    key :

    Returns
    -------

    """
    return fau_color_dict[color]


def adjust_color(key: str, amount: Optional[float] = 1.5) -> str:
    import colorsys
    import matplotlib.colors as mc

    c = colorsys.rgb_to_hls(*mc.to_rgb(fau_color(key)))
    return mc.to_hex(colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2]))
