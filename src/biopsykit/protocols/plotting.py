"""Module providing some standard plots for visualizing data collected during a psychological protocol."""
from typing import Union, Dict, Optional, Sequence, Tuple, Any, Iterable

import re

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.ticker as mticks
from matplotlib.legend_handler import HandlerTuple

import biopsykit.colors as colors
from biopsykit.plotting import lineplot, multi_feature_boxplot, feature_boxplot
from biopsykit.protocols._utils import _get_sample_times
from biopsykit.utils.exceptions import ValidationError
from biopsykit.utils.data_processing import get_subphase_durations
from biopsykit.utils.datatype_helper import (
    is_saliva_mean_se_dataframe,
    is_saliva_raw_dataframe,
    SalivaRawDataFrame,
    SalivaMeanSeDataFrame,
    MergedStudyDataDict,
    MeanSeDataFrame,
    is_mean_se_dataframe,
    SalivaFeatureDataFrame,
    is_saliva_feature_dataframe,
)

_hr_ensemble_plot_params = {
    "figsize": (12, 5),
    "colormap": colors.fau_palette_blue("ensemble"),
    "linestyles": ["-", "--", ":", "-."],
    "ensemble_alpha": 0.4,
    "background_color": ["#e0e0e0", "#9e9e9e", "#808080", "#6b6b6b"],
    "background_alpha": 0.5,
    "xlabel": r"Time [s]",
    "xaxis_minor_tick_locator": mticks.MultipleLocator(60),
    "ylabel": r"$\Delta$HR [%]",
    "legend_loc": "lower right",
    "legend_bbox_to_anchor": (0.99, 0.01),
    "phase_text": "{}",
    "end_phase_text": "End {}",
    "end_phase_line_color": "#e0e0e0",
    "end_phase_line_style": "--",
    "end_phase_line_width": 2.0,
}

_hr_mean_plot_params = {
    "figsize": (12, 5),
    "colormap": colors.fau_palette_blue("line_2"),
    "linestyles": ["-", "--"],
    "markers": ["o", "P"],
    "background_color": ["#e0e0e0", "#bdbdbd", "#9e9e9e"],
    "background_alpha": 0.5,
    "x_offset": 0.1,
    "ylabel": r"$\Delta$HR [%]",
    "phase_text": "{}",
}

_saliva_feature_params: Dict[str, Dict[str, Any]] = {
    "ylabel": {
        "cortisol": {
            "auc": r"Cortisol AUC $\left[\frac{nmol \cdot min}{l} \right]$",
            "auc_g": r"Cortisol AUC $\left[\frac{nmol \cdot min}{l} \right]$",
            "auc_i": r"Cortisol AUC $\left[\frac{nmol \cdot min}{l} \right]$",
            "slope": r"Cortisol Change $\left[\frac{nmol}{l \cdot min} \right]$",
            "max": r"Cortisol $\left[\frac{nmol}{l} \right]$",
            "max_inc": r"Cortisol $\left[\frac{nmol}{l} \right]$",
        },
        "amylase": {
            "auc_g": r"Amylase AUC $\left[\frac{U \cdot min}{l} \right]$",
            "auc_i": r"Amylase AUC $\left[\frac{U \cdot min}{l} \right]$",
            "slope": r"Amylase Change $\left[\frac{U}{l \cdot min} \right]$",
            "max": r"Amylase $\left[\frac{U}{l} \right]$",
            "max_inc": r"Amylase $\left[\frac{U}{l} \right]$",
        },
    },
    "xticklabels": {
        "auc": r"$AUC_{$}$",
        "auc_g": r"$AUC_G$",
        "auc_i": r"$AUC_I$",
        "slope": r"$a_{§}$",
        "max_inc": r"$\Delta c_{max}$",
        "cmax": r"$c_{max}$",
    },
}

_saliva_plot_params: Dict = {
    "colormap": colors.fau_palette_blue("line_2"),
    "linestyles": ["-", "--"],
    "markers": ["o", "P"],
    "test_title": "",
    "test_color": "#9e9e9e",
    "test_alpha": 0.5,
    "multi_x_offset": 0.01,
    "xlabel": "Time [min]",
    "ylabel": {
        "cortisol": "Cortisol [nmol/l]",
        "amylase": "sAA [U/l]",
        "il6": "IL-6 [pg/ml]",
    },
    "legend_title": {"cortisol": "Cortisol", "amylase": "sAA", "il6": "IL-6"},
}


def hr_ensemble_plot(
    data: MergedStudyDataDict,
    subphases: Optional[Dict[str, Dict[str, int]]] = None,
    **kwargs,
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    r"""Draw a heart rate ensemble plot.

    This function plots time-series heart rate continuously as ensemble plot (mean ± standard error).
    If the data consist of multiple phases, data from each phase are overlaid in the same plot.
    If each phase additionally consists of subphases, the single subphases are highlighted in the plot.

    The input data is expected to be a :obj:`~biopsykit.utils.datatype_helper.MergedStudyDataDict`, i.e.,
    a dictionary with merged time-series heart rate data, of multiple subjects, split into individual phases.
    Per phase, the data of each subjects have same length and are combined into one common dataframe.


    Parameters
    ----------
    data : :obj:`~biopsykit.utils.datatype_helper.MergedStudyDataDict`
        dict with heart rate data to plot
    subphases : dict, optional
        dictionary with phases (keys) and subphases (values - dict with subphase names and subphase durations) or
        ``None`` if no subphases are present. Default: ``None``
    **kwargs : dict, optional
        optional arguments for plot configuration.

        To style general plot appearance:

        * ``ax``: pre-existing axes for the plot. Otherwise, a new figure and axes object is created and returned.
        * ``colormap``: colormap to plot data from different phases
        * ``ensemble_alpha``: transparency value for ensemble plot errorband (around mean). Default: 0.4
        * ``linestyles``: list of line styles for ensemble plots. Must match the number of phases to plot
        * ``phase_text``: string pattern to customize phase name shown in legend with placeholder for subphase name.
          Default: "{}"

        To style axes:

        * ``xlabel``: label of x axis. Default: ":math:`Time [s]`"
        * ``xaxis_minor_tick_locator``: locator object to style x axis minor ticks. Default: 60 sec
        * ``ylabel``: label of y axis. Default: ":math:`\Delta HR [\%]`"
        * ``ylims``: y axis limits. Default: ``None`` to automatically infer limits

        To style the annotations at the end of each phase:

        * ``end_phase_text``: string pattern to customize text at the end of phase with placeholder for phase name.
          Default: "{}"
        * ``end_phase_line_color``: line color of vertical lines used to indicate end of phase. Default: "#e0e0e0"
        * ``end_phase_line_width``: line width of vertical lines used  to indicate end of phase. Default: 2.0

        To style legend:

        * ``legend_loc``: location of legend. Default: "lower right"
        * ``legend_bbox_to_anchor``: box that is used to position the legend in conjunction with ``legend_loc``


    Returns
    -------
    fig : :class:`~matplotlib.figure.Figure`
        figure object
    ax : :class:`~matplotlib.axes.Axes`
        axes object


    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.MergedStudyDataDict`
        dictionary format
    :func:`~biopsykit.utils.data_processing.merge_study_data_dict`
        function to build ``MergedStudyDataDict``


    Examples
    --------
    >>> from biopsykit.protocols.plotting import hr_ensemble_plot
    >>> # Example with subphases
    >>> subphase_dict = {
    >>>     "Phase1": {"Baseline": 60, "Stress": 120, "Recovery": 60},
    >>>     "Phase2": {"Baseline": 60, "Stress": 120, "Recovery": 60},
    >>>     "Phase3": {"Baseline": 60, "Stress": 120, "Recovery": 60}
    >>> }
    >>> fig, ax = hr_ensemble_plot(data=data, subphases=subphase_dict)

    """
    ax: plt.Axes = kwargs.pop("ax", None)
    if ax is None:
        figsize = kwargs.get("figsize", _hr_ensemble_plot_params.get("figsize"))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    colormap = kwargs.get("colormap", _hr_ensemble_plot_params.get("colormap"))
    sns.set_palette(colormap)
    linestyles = kwargs.get("linestyles", _hr_ensemble_plot_params.get("linestyles"))

    xlabel = kwargs.get("xlabel", _hr_ensemble_plot_params.get("xlabel"))
    ylabel = kwargs.get("ylabel", _hr_ensemble_plot_params.get("ylabel"))
    ylims = kwargs.get("ylims", _hr_ensemble_plot_params.get("ylims"))
    xaxis_minor_tick_locator = kwargs.get(
        "xaxis_minor_tick_locator", _hr_ensemble_plot_params.get("xaxis_minor_tick_locator")
    )

    ensemble_alpha = kwargs.get("ensemble_alpha", _hr_ensemble_plot_params.get("ensemble_alpha"))

    phase_text = kwargs.get("phase_text", _hr_ensemble_plot_params.get("phase_text"))

    legend_loc = kwargs.get("legend_loc", _hr_ensemble_plot_params.get("legend_loc"))
    legend_bbox_to_anchor = kwargs.get("legend_bbox_to_anchor", _hr_ensemble_plot_params.get("legend_bbox_to_anchor"))

    for i, phase in enumerate(data):
        df_hr_phase = data[phase]
        x = df_hr_phase.index
        hr_mean = df_hr_phase.mean(axis=1)
        hr_stderr = df_hr_phase.std(axis=1) / np.sqrt(df_hr_phase.shape[1])
        ax.plot(x, hr_mean, zorder=2, label=phase_text.format(phase), linestyle=linestyles[i])
        ax.fill_between(x, hr_mean - hr_stderr, hr_mean + hr_stderr, zorder=1, alpha=ensemble_alpha)
        _hr_ensemble_plot_end_phase_annotation(ax, df_hr_phase, phase, i, **kwargs)

    if subphases is not None:
        _hr_ensemble_plot_subphase_vspans(ax, data, subphases, **kwargs)

    ax.set_xlabel(xlabel)
    ax.xaxis.set_minor_locator(xaxis_minor_tick_locator)
    ax.tick_params(axis="x", which="both", bottom=True)

    ax.set_ylabel(ylabel)
    ax.tick_params(axis="y", which="major", left=True)

    if ylims is not None:
        ax.margins(x=0)
        ax.set_ylim(ylims)
    else:
        ax.margins(0, 0.1)

    ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)

    fig.tight_layout()
    return fig, ax


def _hr_ensemble_plot_end_phase_annotation(ax: plt.Axes, data: pd.DataFrame, phase: str, i: int, **kwargs):
    """Add End Phase annotations to heart rate ensemble plot.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        axes object
    data : :class:`~pandas.DataFrame`
        data belonging to ``phase``
    phase : str
        phase to add annotations
    i : int
        counter of phase

    """
    end_phase_text = kwargs.get("end_phase_text", _hr_ensemble_plot_params.get("end_phase_text"))
    end_phase_line_color = kwargs.get("end_phase_line_color", _hr_ensemble_plot_params.get("end_phase_line_color"))
    end_phase_line_style = kwargs.get("end_phase_line_style", _hr_ensemble_plot_params.get("end_phase_line_style"))
    end_phase_line_width = kwargs.get("end_phase_line_width", _hr_ensemble_plot_params.get("end_phase_line_width"))

    ax.vlines(
        x=len(data),
        ymin=0,
        ymax=1,
        transform=ax.get_xaxis_transform(),
        ls=end_phase_line_style,
        lw=end_phase_line_width,
        colors=end_phase_line_color,
        zorder=3,
    )
    ax.annotate(
        text=end_phase_text.format(phase),
        xy=(len(data), 0.85 - 0.05 * i),
        xytext=(-5, 0),
        xycoords=ax.get_xaxis_transform(),
        textcoords="offset points",
        ha="right",
        fontsize="small",
        bbox=dict(facecolor="#e0e0e0", alpha=0.7, boxstyle="round"),
        zorder=3,
    )


def _hr_ensemble_plot_subphase_vspans(
    ax: plt.Axes, data: Dict[str, pd.DataFrame], subphases: Dict[str, Dict[str, int]], **kwargs
):
    """Add subphase vertical spans (vspans) to heart rate ensemble plot.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        axes object
    data : :class:`~pandas.DataFrame`
        data belonging to ``phase``
    subphases : dict
        dictionary with phases (keys) and subphases (values - dict with subphase names and subphase durations)

    """
    num_phases = len(data.keys())

    bg_colors = kwargs.get("background_color", _hr_ensemble_plot_params.get("background_color"))
    bg_alphas = kwargs.get("background_alpha", _hr_ensemble_plot_params.get("background_alpha"))
    bg_alphas = [bg_alphas] * num_phases

    subphase_times = [get_subphase_durations(df, subphases[phase]) for phase, df in data.items()]
    subphase_times = np.array(subphase_times)
    subphase_times = np.max(subphase_times, axis=0)

    subphase_names = np.array([list(subphase_dict.keys()) for phase, subphase_dict in subphases.items()])

    if not (subphase_names[0] == subphase_names).all():
        raise ValueError("Subphases must be the same for all phases!")
    subphase_names = subphase_names[0]

    for i, subphase in enumerate(subphase_names):
        start, end = subphase_times[i]
        color = bg_colors[i]
        alpha = bg_alphas[i]
        ax.axvspan(start, end, color=color, alpha=alpha, zorder=0, lw=0)
        ax.text(
            x=start + 0.5 * (end - start),
            y=0.95,
            transform=ax.get_xaxis_transform(),
            s=subphase,
            ha="center",
            va="center",
        )
    p = mpatch.Rectangle(
        xy=(0, 0.9),
        width=1,
        height=0.1,
        transform=ax.transAxes,
        color="white",
        alpha=0.4,
        zorder=3,
        lw=0,
    )
    ax.add_patch(p)
    ax.set_xticks([start for (start, end) in subphase_times])


def hr_mean_plot(
    data: MeanSeDataFrame,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    r"""Plot course of heart rate as mean ± standard error over phases (and subphases) of a psychological protocol.

    The correct plot is automatically inferred from the provided data:

    * only ``phase`` index level: plot phases over x axis
    * ``phase`` and ``subphase`` index levels: plot subphases over x axis, highlight phases as vertical spans
    * additionally: ``condition`` level: plot data of different conditions individually
      (corresponds to ``hue`` parameter in :func:`~biopsykit.plotting.lineplot`)

    Parameters
    ----------
    data : :class:`~biopsykit.utils.datatype_helper.MeanSeDataFrame`
        Heart rate data to plot. Must be provided as ``MeanSeDataFrame`` with columns ``mean`` and ``se``
        computed over phases (and, if available, subphases)
    **kwargs
        additional parameters to be passed to the plot, such as:

        * ``ax``: pre-existing axes for the plot. Otherwise, a new figure and axes object is created and returned.
        * ``colormap``: colormap to plot data from different phases
        * ``figsize``: tuple specifying figure dimensions
        * ``x_offset``: offset value to move different groups along the x axis for better visualization.
          Default: 0.05
        * ``xlabel``: label of x axis. Default: "Subphases" (if subphases are present)
          or "Phases" (if only phases are present).
        * ``ylabel``: label of y axis. Default: ":math:`\Delta HR [%]`"
        * ``ylims``: list to manually specify y axis limits, float to specify y axis margin
          (see :meth:`~matplotlib.axes.Axes.margins()` for further information), or ``None`` to automatically infer
          y axis limits.
        * ``marker``: string or list of strings to specify marker style.
          If ``marker`` is a string, then marker of each line will have the same style.
          If ``marker`` is a list, then marker of each line will have a different style.
        * ``linestyle``: string or list of strings to specify line style.
          If ``linestyle`` is a string, then each line will have the same style.
          If ``linestyle`` is a list, then each line will have a different style.

    Returns
    -------
    fig : :class:`~matplotlib.figure.Figure`
        figure object
    ax : :class:`~matplotlib.axes.Axes`
        axes object

    See Also
    --------
    :func:`~biopsykit.plotting.lineplot`
        Plot data as lineplot with mean and standard error

    """
    fig, ax = _plot_get_fig_ax(**kwargs)

    kwargs.update({"ax": ax})

    # get all plot parameter
    sns.set_palette(kwargs.get("colormap", _hr_mean_plot_params.get("colormap")))

    ylabel = kwargs.get("ylabel", _hr_mean_plot_params.get("ylabel"))
    ylims = kwargs.get("ylims", None)

    phase_dict = _hr_mean_get_phases_subphases(data)
    num_phases = len(phase_dict)
    num_subphases = [len(arr) for arr in phase_dict.values()]
    x_vals = _hr_mean_get_x_vals(num_phases, num_subphases)

    # build x axis, axis limits and limits for phase spans
    dist = np.mean(np.ediff1d(x_vals))
    x_lims = np.append(x_vals, x_vals[-1] + dist)
    x_lims = x_lims - 0.5 * np.ediff1d(x_lims, to_end=dist)

    if "condition" in data.index.names:
        for i, (key, df) in enumerate(data.groupby("condition")):
            _hr_mean_plot(df, x_vals, key, index=i, **kwargs)
    else:
        _hr_mean_plot(data, x_vals, "Data", index=0, **kwargs)

    # add decorators to phases if subphases are present
    if sum(num_subphases) > 0:
        _hr_mean_plot_subphase_annotations(phase_dict, x_lims, **kwargs)

    # customize x axis
    ax.tick_params(axis="x", bottom=True)
    ax.set_xticks(x_vals)
    ax.set_xlim(np.min(x_lims), np.max(x_lims))
    _hr_mean_style_x_axis(ax, phase_dict, num_subphases)

    # customize y axis
    ax.tick_params(axis="y", which="major", left=True)
    ax.set_ylabel(ylabel)

    _hr_mean_plot_set_axis_lims(ylims, ax)

    # customize legend
    if "condition" in data.index.names:
        _hr_mean_add_legend(**kwargs)

    fig.tight_layout()
    return fig, ax


def _hr_mean_plot_set_axis_lims(ylims: Union[Sequence[float], float], ax: plt.Axes):
    if isinstance(ylims, (tuple, list)):
        ax.set_ylim(ylims)
    else:
        ymargin = 0.15
        if isinstance(ylims, float):
            ymargin = ylims
        ax.margins(y=ymargin)
    ax.margins(x=0)
    ax.relim()


def _hr_mean_plot(data: MeanSeDataFrame, x_vals: np.array, key: str, index: int, **kwargs):
    ax: plt.Axes = kwargs.get("ax")
    x_offset = kwargs.get("x_offset", _hr_mean_plot_params.get("x_offset"))
    markers = kwargs.get("markers", _hr_mean_plot_params.get("markers"))
    linestyles = kwargs.get("linestyles", _hr_mean_plot_params.get("linestyles"))

    if isinstance(markers, list):
        markers = markers[index]

    if isinstance(linestyles, list):
        linestyles = linestyles[index]

    is_mean_se_dataframe(data)
    if isinstance(data.columns, pd.MultiIndex):
        # if data has multiindex columns: drop all levels except the last one
        # (which is expected to contain ["mean", "se"])
        data.columns = data.columns.droplevel(list(range(0, data.columns.nlevels - 1)))

    ax.errorbar(
        x=x_vals + index * x_offset,
        y=data["mean"],
        label=key,
        yerr=data["se"],
        capsize=3,
        marker=markers,
        linestyle=linestyles,
    )


def _hr_mean_add_legend(**kwargs):
    """Add legend to mean HR plot."""
    ax: plt.Axes = kwargs.get("ax")
    legend_loc = kwargs.get("legend_loc", "upper left")
    # get handles
    handles, labels = ax.get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    # use them in the legend
    if legend_loc == "upper left":
        bbox_to_anchor = (0.01, 0.90)
    elif legend_loc == "upper right":
        bbox_to_anchor = (0.99, 0.90)
    else:
        bbox_to_anchor = None
    ax.legend(
        handles,
        labels,
        loc=legend_loc,
        bbox_to_anchor=bbox_to_anchor,
        numpoints=1,
    )


def _hr_mean_style_x_axis(ax: plt.Axes, phase_dict: Dict[str, Sequence[str]], num_subphases: Sequence[int], **kwargs):
    """Style x axis of mean HR plot.

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        axes object
    phase_dict : dict
        dictionary with phase names (keys) and dict of subphases (values)
    num_subphases : list
        list with number of subphases for each phase

    """
    if sum(num_subphases) == 0:
        # no subphases
        ax.set_xticklabels(phase_dict.keys())
        ax.set_xlabel(kwargs.get("xlabel", "Phases"))
    else:
        ax.set_xticklabels([s for subph in phase_dict.values() for s in subph])
        ax.set_xlabel(kwargs.get("xlabel", "Subphases"))


def _hr_mean_plot_subphase_annotations(phase_dict: Dict[str, Sequence[str]], xlims: Sequence[float], **kwargs):
    """Add subphase annotations to mean HR plot.

    Parameters
    ----------
    phase_dict : dict
        dictionary with phase names (keys) and dict of subphases (values)
    xlims : list
        x axis limits

    """
    ax: plt.Axes = kwargs.get("ax")
    bg_colors = kwargs.get("background_color", _hr_mean_plot_params.get("background_color"))
    bg_alpha = kwargs.get("background_alpha", _hr_mean_plot_params.get("background_alpha"))
    phase_text = kwargs.get("phase_text", _hr_mean_plot_params.get("phase_text"))

    num_phases = len(phase_dict)
    num_subphases = [len(arr) for arr in phase_dict.values()]

    x_spans = _hr_mean_get_x_spans(num_phases, num_subphases)

    for (i, phase), bg_color in zip(enumerate(phase_dict), bg_colors):
        left, right = x_spans[i]
        ax.axvspan(xlims[left], xlims[right], color=bg_color, alpha=bg_alpha, zorder=0, lw=0)
        name = phase_text.format(phase)
        ax.text(
            x=xlims[left] + 0.5 * (xlims[right] - xlims[left]),
            y=0.95,
            s=name,
            transform=ax.get_xaxis_transform(),
            horizontalalignment="center",
            verticalalignment="center",
        )

    p = mpatch.Rectangle(
        xy=(0, 0.9),
        width=1,
        height=0.1,
        transform=ax.transAxes,
        color="white",
        alpha=0.4,
        zorder=3,
        lw=0,
    )
    ax.add_patch(p)


def _hr_mean_get_x_spans(num_phases: int, num_subphases: Sequence[int]):
    if sum(num_subphases) == 0:
        x_spans = list(zip([0] + list(range(0, num_phases)), list(range(0, num_phases))))
    else:
        x_spans = list(zip([0] + list(np.cumsum(num_subphases)), list(np.cumsum(num_subphases))))
    return x_spans


def _hr_mean_get_x_vals(num_phases: int, num_subphases: Sequence[int]):
    if sum(num_subphases) == 0:
        x_vals = np.linspace(0, 10, num_phases)
    else:
        x_vals = np.linspace(0, 10, sum(num_subphases))
    return x_vals


def _hr_mean_get_phases_subphases(data: pd.DataFrame) -> Dict[str, Sequence[str]]:
    if "condition" in data.index.names:
        data = [value for key, value in data.groupby("condition")][0]

    phases = data.index.get_level_values("phase").unique()

    if "subphase" in data.index.names:
        phase_dict = {phase: list(df.index.get_level_values("subphase")) for phase, df in data.groupby("phase")}
    else:
        phase_dict = {phase: [] for phase in phases}

    return phase_dict


def saliva_plot(
    data: Union[
        SalivaRawDataFrame, SalivaMeanSeDataFrame, Dict[str, SalivaRawDataFrame], Dict[str, SalivaMeanSeDataFrame]
    ],
    saliva_type: Optional[str] = None,
    sample_times: Optional[Union[Sequence[int], Dict[str, Sequence[int]]]] = None,
    test_times: Optional[Sequence[int]] = None,
    sample_times_absolute: Optional[bool] = False,
    **kwargs,
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    r"""Plot saliva data during psychological protocol as mean ± standard error.

    The function accepts raw saliva data per subject (:obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`)
    as well as pre-computed mean and standard error values of saliva samples
    ( :obj:`~biopsykit.utils.datatype_helper.SalivaMeanSeDataFrame`). To combine data from multiple saliva
    types (maximum: 2) into one plot a dict can be passed to ``data``.

    If a psychological test (e.g., TSST, MIST, or Stroop) was performed, the test time is highlighted as vertical span
    within the plot.

    .. note::
        If no sample times are provided (neither via ``time`` column in ``data`` nor via ``sample_times`` parameter)
        then ``samples`` will be used as x axis


    Parameters
    ----------
    data : :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`, \
           :obj:`~biopsykit.utils.datatype_helper.SalivaMeanSeDataFrame`, or dict of such
        Saliva data to plot. Must either be provided as ``SalivaRawDataFrame`` with raw saliva data per subject or
        as ``SalivaMeanSeDataFrame`` with columns ``mean`` and ``se`` computed per saliva sample. To plot data from
        multiple saliva types (maximum: 2) a dict can be passed (keys: saliva types, values: saliva data).
    saliva_type : {"cortisol", "amylase", "il6"}, optional
        saliva type to be plotted. If a dict is passed and ``saliva_type`` is ``None``
        the saliva types are inferred from dict keys.
    sample_times : list or dict of lists
        sample times in minutes relative to psychological test or a dict of such if sample times are different for
        the individual saliva types.
    test_times : list of int, optional
        start and end times of psychological test (in minutes) or ``None`` if no test was performed
    sample_times_absolute : bool, optional
        ``True`` if absolute sample times were provided (i.e., the duration of the psychological test was already
        added to the sample times), ``False`` if relative sample times were provided and absolute times should be
        computed based on test times specified by ``test_times``. Default: ``False``
    **kwargs
        additional parameters to be passed to the plot.

        To style general plot appearance:

        * ``ax``: pre-existing axes for the plot. Otherwise, a new figure and axes object is created and returned.
        * ``colormap``: colormap to plot data from different phases
        * ``figsize``: tuple specifying figure dimensions
        * ``marker``: string or list of strings to specify marker style.
          If ``marker`` is a string, then marker of each line will have the same style.
          If ``marker`` is a list, then marker of each line will have a different style.
        * ``linestyle``: string or list of strings to specify line style.
          If ``linestyle`` is a string, then each line will have the same style.
          If ``linestyle`` is a list, then each line will have a different style.

        To style axes:

        * ``x_offset``: offset value to move different groups along the x axis for better visualization.
          Default: 0.05
        * ``xlabel``: label of x axis. Default: "Subphases" (if subphases are present).
          or "Phases" (if only phases are present)
        * ``ylabel``: label of y axis. Default: ":math:`\Delta HR [%]`"
        * ``ylims``: list to manually specify y axis limits, float to specify y axis margin
          (see :meth:`~matplotlib.axes.Axes.margins()` for further information), or ``None`` to automatically infer
          y axis limits.

        To style the vertical span highlighting the psychological test in the plot:

        * ``test_title``: title of test
        * ``test_color``: color of vspan. Default: #9e9e9e
        * ``test_alpha``: transparency value of vspan: Default: 0.5

    Returns
    -------
    fig : :class:`~matplotlib.figure.Figure`
        figure object
    ax : :class:`~matplotlib.axes.Axes`
        axes object

    See Also
    --------
    :func:`~biopsykit.plotting.lineplot`
        Plot data as lineplot with mean and standard error

    """
    fig, ax = _plot_get_fig_ax(**kwargs)
    kwargs.update({"ax": ax})

    if saliva_type is None and not isinstance(data, dict):
        raise ValueError("If 'saliva_type' is None, you must pass a dict!")

    if isinstance(data, pd.DataFrame):
        # multiple saliva data were passed in a dict => get the selected saliva type
        data = {saliva_type: data}
        sample_times = {saliva_type: sample_times}

    linestyle = kwargs.pop("linestyle", None)
    marker = kwargs.pop("marker", "o")
    colormap = kwargs.pop("colormap", None)
    for i, key in enumerate(data):
        kwargs = _saliva_plot_extract_style_params(key, linestyle, marker, colormap, **kwargs)
        _saliva_plot(
            data=data[key],
            saliva_type=key,
            counter=i,
            sample_times=sample_times[key],
            test_times=test_times,
            sample_times_absolute=sample_times_absolute,
            **kwargs,
        )

    if len(data) > 1:
        saliva_plot_combine_legend(fig, saliva_types=list(data.keys()), **kwargs)
    else:
        fig.tight_layout()

    return fig, ax


def _saliva_plot_extract_style_params(
    key: str,
    linestyle: Union[Dict[str, str], str],
    marker: Union[Dict[str, str], str],
    colormap: Union[Dict[str, str], str],
    **kwargs,
):
    ls = _saliva_plot_get_plot_param(linestyle, key)
    if linestyle is not None:
        kwargs.setdefault("linestyle", ls)

    m = _saliva_plot_get_plot_param(marker, key)
    if marker is not None:
        kwargs.setdefault("marker", m)

    cmap = _saliva_plot_get_plot_param(colormap, key)
    if colormap is not None:
        kwargs.setdefault("colormap", cmap)

    return kwargs


def _saliva_plot_hue_style(data: pd.DataFrame):
    hue = "condition" if "condition" in data.index.names else None
    style = "condition" if "condition" in data.index.names else None
    return hue, style


def _saliva_plot_sanitize_dicts(
    data: Union[Dict[str, pd.DataFrame], pd.DataFrame], ylabel: Union[Dict[str, str], str], saliva_type: str
):
    if isinstance(ylabel, dict):
        ylabel = ylabel[saliva_type]

    if isinstance(data, dict):
        # multiple saliva data were passed in a dict => get the selected saliva type
        data = data[saliva_type]

    return data, ylabel


def _saliva_plot(
    data: Union[SalivaRawDataFrame, SalivaMeanSeDataFrame],
    saliva_type: str,
    counter: int,
    sample_times: Optional[Union[Sequence[int], Dict[str, Sequence[int]]]] = None,
    test_times: Optional[Sequence[int]] = None,
    sample_times_absolute: Optional[bool] = False,
    **kwargs,
):

    ax: plt.Axes = kwargs.get("ax")

    test_times = test_times or [0, 0]
    test_title = kwargs.get("test_title", _saliva_plot_params.get("test_title"))
    test_color = kwargs.get("test_color", _saliva_plot_params.get("test_color"))
    test_alpha = kwargs.get("test_alpha", _saliva_plot_params.get("test_alpha"))
    xlabel = kwargs.get("xlabel", _saliva_plot_params.get("xlabel"))
    ylabel = kwargs.get("ylabel", _saliva_plot_params.get("ylabel"))
    xticks = kwargs.get("xticks")
    xaxis_tick_locator = kwargs.get("xaxis_tick_locator")

    data, ylabel = _saliva_plot_sanitize_dicts(data, ylabel, saliva_type)

    _assert_saliva_data_input(data, saliva_type)

    data = data.copy()

    if sample_times is None and "time" not in data.reset_index().columns:
        x = "sample"
        xlabel = "Sample"
    else:
        sample_times = _get_sample_times(data, sample_times, test_times, sample_times_absolute)
        if "time" in data.index.names:
            data.index = data.index.droplevel("time")
        data["time"] = sample_times * int(len(data) / len(sample_times))
        x = "time"

    hue, style = _saliva_plot_hue_style(data)
    kwargs.setdefault("markers", "o")

    if counter == 0 and len(ax.lines) == 0:
        kwargs.setdefault("colormap", colors.fau_palette_blue("line_2"))
        if sum(test_times) != 0:
            ax.axvspan(*test_times, color=test_color, alpha=test_alpha, zorder=1, lw=0)
            ax.text(
                x=test_times[0] + 0.5 * (test_times[1] - test_times[0]),
                y=0.95,
                transform=ax.get_xaxis_transform(),
                s=test_title,
                fontsize="small",
                horizontalalignment="center",
                verticalalignment="top",
            )
    else:
        kwargs.setdefault("colormap", colors.fau_palette_tech("line_2"))
        # the was already something drawn into the axis => we are using the same axis to add another feature
        ax_twin = ax.twinx()
        kwargs.update({"ax": ax_twin, "show_legend": False})

    kwargs.update({"xlabel": xlabel, "ylabel": ylabel})

    lineplot(data=data, x=x, y=saliva_type, hue=hue, style=style, **kwargs)

    _saliva_plot_style_xaxis(xticks, xaxis_tick_locator, ax)


def _assert_saliva_data_input(data: pd.DataFrame, saliva_type: str):
    ret = is_saliva_raw_dataframe(data, saliva_type, raise_exception=False)
    ret = ret or is_saliva_mean_se_dataframe(data, raise_exception=False)
    if not ret:
        raise ValidationError("'data' is expected to be either a SalivaRawDataFrame or a SalivaMeanSeDataFrame!")


def _saliva_plot_get_plot_param(param: Union[Dict[str, str], str], key: str):
    if isinstance(param, dict):
        p = param[key]
    else:
        p = param
    return p


def _saliva_plot_style_xaxis(xticks: Sequence[str], xaxis_tick_locator: mticks.Locator, ax: plt.Axes):
    if xticks is not None and xaxis_tick_locator is not None:
        ax.xaxis.set_major_locator(xaxis_tick_locator)
        ax.xaxis.set_ticks(xticks)


def saliva_plot_combine_legend(fig: plt.Figure, ax: plt.Axes, saliva_types: Sequence[str], **kwargs):
    """Combine multiple legends of ``saliva_plot`` into one joint legend outside of plot.

    If data from multiple saliva types are combined into one plot (e.g., by calling
    :func:`~biopsykit.protocols.plotting.saliva_plot` on the same plot twice) then two separate legend are created.
    This function can be used to combine the two legends into one.

    Parameters
    ----------
    fig : :class:`~matplotlib.figure.Figure`
        figure object
    ax : :class:`~matplotlib.axes.Axes`
        axes object
    saliva_types : list
        list of saliva types in plot
    **kwargs
        additional arguments to customize plot, such as:

        * ``legend_loc``: Location of legend. Default: ``upper center``
        * ``legend_size``: Legend size. Default: ``small``
        * ``rect``: Rectangle in normalized figure coordinates into which the whole subplots area
          (including labels) will fit. Used to conveniently place legend outside of figure.

    """
    legend_loc = kwargs.get("legend_loc", "upper center")
    legend_size = kwargs.get("legend_size", "small")
    rect = kwargs.get("rect", (0, 0, 1.0, 0.95))
    labels = [ax.get_legend_handles_labels()[1] for ax in fig.get_axes()]

    if all([len(label) == 1 for label in labels]):
        # only one group
        handles = [ax.get_legend_handles_labels()[0] for ax in fig.get_axes()]
        handles = [h[0] for handle in handles for h in handle]
        labels = [_saliva_plot_params.get("legend_title")[b] for b in saliva_types]
        ncol = len(handles)
        fig.legend(
            handles,
            labels,
            loc=legend_loc,
            ncol=ncol,
            prop={"size": legend_size},
        )
    else:
        handles = [ax.get_legend_handles_labels()[0] for ax in fig.get_axes()]
        handles = [h[0] for handle in handles for h in handle]
        labels = [ax.get_legend_handles_labels()[1] for ax in fig.get_axes()]
        labels = [
            "{}: {}".format(_saliva_plot_params.get("legend_title")[b], " - ".join(l))
            for b, l in zip(saliva_types, labels)
        ]
        ncol = len(handles)

        fig.legend(
            list(zip(handles[::2], handles[1::2])),
            labels,
            loc=legend_loc,
            ncol=ncol,
            numpoints=1,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            prop={"size": legend_size},
        )
    ax.legend().remove()
    fig.tight_layout(pad=1.0, rect=rect)


def saliva_feature_boxplot(
    data: SalivaFeatureDataFrame,
    x: str,
    saliva_type: str,
    feature: Optional[str] = None,
    stats_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Draw a boxplot with significance brackets, specifically designed for saliva features.

    This is a wrapper of :func:`~biopsykit.plotting.feature_boxplot` that can be used to plot saliva features and
    allows to easily add significance brackets that indicate statistical significance.

    .. note::
        The input data is assumed to be in long-format.


    Parameters
    ----------
    data : :obj:`~biopsykit.utils.datatype_helper.SalivaFeatureDataFrame`
        data to plot
    x : str
        column of x axis in ``data``
    saliva_type : str
        type of saliva data to plot
    feature : str, optional
        name of feature to plot or ``None``
    stats_kwargs : dict, optional
        dictionary with arguments for significance brackets
    **kwargs
        additional arguments that are passed to :func:`~biopsykit.plotting.feature_boxplot` and :func:`~seaborn.boxplot`


    Returns
    -------
    fig : :class:`~matplotlib.figure.Figure`
        figure object
    ax : :class:`~matp~lotlib.axes.Axes`
        axes object


    See Also
    --------
    :func:`~biopsykit.plotting.feature_boxplot`
        plot features as boxplot
    :class:`~biopsykit.stats.stats.StatsPipeline`
        class to create statistical analysis pipelines and get parameter for plotting significance brackets

    """
    is_saliva_feature_dataframe(data, saliva_type)
    if feature is not None:
        if isinstance(feature, str):
            feature = [feature]
        ylabel = _saliva_feature_boxplot_get_ylabels(saliva_type, feature)
        ylabel = [ylabel[f] for f in feature]
        if len(set(ylabel)) == 1:
            kwargs["ylabel"] = ylabel[0]
    return feature_boxplot(data=data, x=x, y=saliva_type, stats_kwargs=stats_kwargs, **kwargs)


def saliva_multi_feature_boxplot(
    data: SalivaFeatureDataFrame,
    saliva_type: str,
    features: Union[Sequence[str], Dict[str, Union[str, Sequence[str]]]],
    hue: Optional[str] = None,
    stats_kwargs: Optional[Dict] = None,
    **kwargs,
) -> Tuple[plt.Figure, Iterable[plt.Axes]]:
    """Draw multiple features as boxplots with significance brackets, specifically designed for saliva features.

    This is a wrapper of :func:`~biopsykit.plotting.multi_feature_boxplot` that can be used to plot saliva features and
    allows to easily add significance brackets that indicate statistical significance.

    .. note::
        The input data is assumed to be in long-format.

    Parameters
    ----------
    data : :obj:`~biopsykit.utils.datatype_helper.SalivaFeatureDataFrame`
        data to plot
    saliva_type : str
        type of saliva data to plot
    hue : str, optional
        column name of grouping variable. Default: ``None``
    features : list of str or dict of str
        features to plot. If ``features`` is a list, each entry must correspond to one feature category in the
        index level specified by ``group``. A separate subplot will be created for each feature.
        If similar features (i.e., different `slope` or `AUC` parameters) should be combined into one subplot,
        ``features`` can be provided as dictionary.
        Then, the dict keys specify the feature category (a separate subplot will be created for each category)
        and the dict values specify the feature (or list of features) that are combined into the subplots.
    stats_kwargs : dict, optional
        nested dictionary with arguments for significance brackets.
        See :func:`~biopsykit.plotting.feature_boxplot` for further information


    Returns
    -------
    fig : :class:`~matplotlib.figure.Figure`
        figure object
    axs : list of :class:`matplotlib.axes.Axes`
        list of subplot axes objects


    See Also
    --------
    :func:`~biopsykit.plotting.multi_feature_boxplot`
        plot multiple features as boxplots
    :class:`~biopsykit.stats.stats.StatsPipeline`
        class to create statistical analysis pipelines and get parameter for plotting significance brackets

    """
    x = kwargs.pop("x", "saliva_feature")

    if isinstance(features, str):
        # ensure list
        features = [features]

    kwargs.setdefault("xticklabels", _saliva_feature_boxplot_get_xticklabels(features))
    kwargs.setdefault("ylabels", _saliva_feature_boxplot_get_ylabels(saliva_type, features))

    return multi_feature_boxplot(
        data,
        x=x,
        y=saliva_type,
        group="saliva_feature",
        features=features,
        hue=hue,
        stats_kwargs=stats_kwargs,
        **kwargs,
    )


def _saliva_feature_boxplot_get_xticklabels(features: Dict[str, str]) -> Dict[str, Sequence[str]]:
    xlabel_dict = {}
    for feature in features:
        cols = features[feature]
        if isinstance(cols, str):
            cols = [cols]
        labels = []
        for c in cols:
            if "slope" in c:
                label = _saliva_feature_params["xticklabels"]["slope"].replace("§", re.findall(r"slope(\w+)", c)[0])
            else:
                label = _saliva_feature_params["xticklabels"][c]
            labels.append(label)

        xlabel_dict[feature] = labels
    return xlabel_dict


def _saliva_feature_boxplot_get_ylabels(saliva_type: str, features: Union[str, Sequence[str]]) -> Dict[str, str]:
    ylabels = _saliva_feature_params["ylabel"][saliva_type]
    if isinstance(features, str):
        features = [features]
    for feature in features:
        if "slope" in feature:
            ylabels[feature] = ylabels["slope"]

    return ylabels


def _plot_get_fig_ax(**kwargs):
    ax: plt.Axes = kwargs.get("ax", None)
    if ax is None:
        figsize = kwargs.get("figsize", _hr_mean_plot_params.get("figsize"))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax
