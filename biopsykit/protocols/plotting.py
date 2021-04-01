"""Provides some standard plots for plotting biopsychological data during a psychological protocol"""
from typing import Union, Dict, Optional, Sequence, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import biopsykit.colors as colors
import matplotlib.patches as mpatch

_saliva_params: Dict = {
    "colormap": colors.cmap_fau_blue("2_lp"),
    "line_styles": ["-", "--"],
    "markers": ["o", "P"],
    # 'background_color': "#e0e0e0",
    # 'background_alpha': 0.5,
    # 'x_padding': 0.1,
    "test_text": "",
    "test_color": "#9e9e9e",
    "test_alpha": 0.5,
    "x_offsets": [0, 0.5],
    "fontsize": 14,
    "multi_x_offset": 1,
    "multi_fontsize": 10,
    "multi_legend_offset": 0.3,
    "multi_colormap": colors.cmap_fau_phil("2_lp"),
    "xaxis_tick_locator": plt.MultipleLocator(20),
    "xaxis_label": "Time [min]",
    "yaxis_label": {
        "cortisol": "Cortisol [nmol/l]",
        "amylase": "Amylase [U/l]",
        "il6": "IL-6 [pg/ml]",
    },
}

_hr_mean_plot_params = {
    "colormap": colors.cmap_fau_blue("2_lp"),
    "line_styles": ["-", "--"],
    "markers": ["o", "P"],
    "background_color": None,
    "background_alpha": None,
    "x_offsets": [0, 0.05],
    "fontsize": 14,
    "xaxis_label": "Phases",
    "yaxis_label": "Value",
}

_saliva_feature_params = {
    "yaxis_label": {
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
        "max_inc": "$\Delta c_{max}$",
        "cmax": "$c_{max}$",
    },
}


# TODO add support for groups in one dataframe (indicated by group column)
def hr_mean_plot(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    phases: Optional[Sequence[str]] = None,
    subphases: Optional[Sequence[str]] = None,
    groups: Optional[Sequence[str]] = None,
    group_col: Optional[str] = None,
    plot_params: Optional[Dict] = None,
    **kwargs
) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
    """
    Plots the course of heart rate during the complete MIST (mean ± standard error per subphase).

    In case of only one group a pandas dataframe can be passed.

    In case of multiple groups either a dictionary of pandas dataframes can be passed, where each dataframe belongs
    to one group, or one dataframe with a column indicating group membership (parameter ``group_col``).

    Regardless of the kind of input the dataframes need to be in the format of a 'mse dataframe', as returned
    by ``MIST.hr_course_mist`` (see ``MIST.hr_course_mist`` for further information).



    Parameters
    ----------
    data : dataframe or dict
        Heart rate data to plot. Can either be one dataframe (in case of only one group or in case of
        multiple groups, together with `group_col`) or a dictionary of dataframes,
        where one dataframe belongs to one group
    phases : list of str, optional
        list of phase names or ``None`` to infer from data
    subphases : list of str, optional
        list of subphase names of ``None`` to infer from data
    groups : list, optional:
         List of group names. If ``None`` is passed, the groups and their order are inferred from the
         dictionary keys or from the unique values in `group_col`. If list is supplied the groups are
         plotted in that order.
         Default: ``None``
    group_col : str, optional
        Name of group column in the dataframe in case of multiple groups and one dataframe
    plot_params : dict, optional
        dict with adjustable parameters specific for this plot or ``None`` to keep default parameter values.
        For an overview of parameters and their default values, see `mist.hr_course_params`
    kwargs: dict, optional
        optional parameters to be passed to the plot, such as:
            * Axes to plot on, otherwise create a new one. Default: ``None``
            * figsize: tuple specifying figure dimensions
            * ylims: list to manually specify y-axis limits, float to specify y-axis margin (see ``Axes.margin()``
            for further information), None to automatically infer y-axis limits

    Returns
    -------
    tuple or none
        Tuple of Figure and Axes or None if Axes object was passed
    """

    fig: Union[plt.Figure, None] = None
    ax: Union[plt.Axes, None] = kwargs.get("ax", None)
    if ax is None:
        figsize = kwargs.get("figsize", plt.rcParams["figure.figsize"])
        fig, ax = plt.subplots(figsize=figsize)

    hr_mean_plot_params = _hr_mean_plot_params.copy()

    # update default parameter if plot parameter were passed
    if plot_params:
        hr_mean_plot_params.update(plot_params)

    ylims = kwargs.get("ylims", None)

    # get all plot parameter
    sns.set_palette(hr_mean_plot_params["colormap"])
    line_styles = hr_mean_plot_params["line_styles"]
    markers = hr_mean_plot_params["markers"]
    bg_colors = kwargs.get("background_color", hr_mean_plot_params["background_color"])
    bg_alphas = kwargs.get("background_alpha", hr_mean_plot_params["background_alpha"])
    x_offsets = hr_mean_plot_params["x_offsets"]
    fontsize = kwargs.get("fontsize", hr_mean_plot_params["fontsize"])
    xaxis_label = kwargs.get("xlabel", hr_mean_plot_params["xaxis_label"])
    yaxis_label = kwargs.get("ylabel", hr_mean_plot_params["yaxis_label"])
    phase_text = hr_mean_plot_params.get("phase_text", None)

    if phases is None:
        if isinstance(data, pd.DataFrame):
            if isinstance(data.index, pd.MultiIndex):
                phases = list(data.index.get_level_values(0).unique())
            else:
                phases = list(data.index)
        else:
            data_grp = list(data.values())[0]
            if isinstance(data_grp.index, pd.MultiIndex):
                phases = list(data_grp.index.get_level_values(0).unique())
            else:
                phases = list(data_grp.index)

    if subphases is None:
        if isinstance(data, pd.DataFrame):
            if isinstance(data.index, pd.MultiIndex):
                subphases = list(data.index.get_level_values(1).unique())
            else:
                # No subphases
                subphases = [""]
        else:
            data_grp = list(data.values())[0]
            if isinstance(data_grp.index, pd.MultiIndex):
                subphases = list(data_grp.index.get_level_values(1).unique())
            else:
                # No subphases
                subphases = [""]

    if isinstance(data, pd.DataFrame):
        if isinstance(data.index, pd.MultiIndex):
            # get subphase labels from data
            subphase_labels = data.index.get_level_values(1)
            if "condition" in data.index.names:
                groups = list(data.index.get_level_values(level="condition").unique())
        else:
            subphase_labels = [""]
    else:
        if not groups:
            # get group names from dict if groups were not supplied
            groups = list(data.keys())
        # get subphase labels from data
        data_grp = list(data.values())[0]
        if isinstance(data_grp.index, pd.MultiIndex):
            subphase_labels = data_grp.index.get_level_values(1)
        else:
            subphase_labels = [""]

    num_subph = len(subphases)

    # build x axis, axis limits and limits for MIST phase spans
    if len(subphases) == 1:
        x = np.arange(len(phases))
    else:
        x = np.arange(len(subphase_labels))

    xlims = np.append(x, x[-1] + 1)
    xlims = xlims[::num_subph] + 0.5 * (xlims[::num_subph] - xlims[::num_subph] - 1)
    span_lims = [(x_l, x_u) for x_l, x_u in zip(xlims, xlims[1::])]

    # plot data as errorbar with mean and se
    if groups:
        for group, x_off, marker, ls in zip(groups, x_offsets, markers, line_styles):
            ax.errorbar(
                x=x + x_off,
                y=data[group]["mean"],
                label=group,
                yerr=data[group]["se"],
                capsize=3,
                marker=marker,
                linestyle=ls,
            )
    else:
        ax.errorbar(
            x=x,
            y=data["mean"],
            yerr=data["se"],
            capsize=3,
            marker=markers[0],
            linestyle=line_styles[0],
        )

    # add decorators if specified: spans and Phase labels
    if bg_colors is not None:
        for (i, name), (x_l, x_u), color, alpha in zip(
            enumerate(phases), span_lims, bg_colors, bg_alphas
        ):
            ax.axvspan(x_l, x_u, color=color, alpha=alpha, zorder=0, lw=0)
            if phase_text is not None:
                name = phase_text.format(i + 1)

            ax.text(
                x=x_l + 0.5 * (x_u - x_l),
                y=0.95,
                s=name,
                transform=ax.get_xaxis_transform(),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=fontsize,
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

    # customize x axis
    ax.tick_params(axis="x", bottom=True)
    ax.set_xticks(x)
    if len(subphase_labels) == len(x):
        ax.set_xticklabels(subphase_labels)
    else:
        # no subphases
        ax.set_xticklabels(phases)

    xlims = kwargs.get("xlims", [span_lims[0][0], span_lims[-1][-1]])
    ax.set_xlim(xlims)
    ax.set_xlabel(xaxis_label, fontsize=fontsize)

    # customize y axis
    ax.tick_params(axis="y", which="major", left=True)
    if isinstance(ylims, (tuple, list)):
        ax.set_ylim(ylims)
    else:
        ymargin = 0.15
        if isinstance(ylims, float):
            ymargin = ylims
        ax.margins(x=0, y=ymargin)

    ax.set_ylabel(yaxis_label, fontsize=fontsize)

    # axis tick label fontsize
    ax.tick_params(labelsize=fontsize)

    # customize legend
    if groups:
        # get handles
        handles, labels = ax.get_legend_handles_labels()
        # remove the errorbars
        handles = [h[0] for h in handles]
        # use them in the legend
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.90),
            numpoints=1,
            prop={"size": fontsize},
        )

    if fig:
        fig.tight_layout()
        return fig, ax


def saliva_plot(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    biomarker: str,
    saliva_times: Sequence[int],
    test_times: Sequence[int],
    groups: Optional[Sequence[str]] = None,
    group_col: Optional[str] = None,
    **kwargs
) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
    """
    TODO: add documentation

    Parameters
    ----------
    data
    biomarker
    saliva_times
    test_times
    groups
    group_col
    kwargs: dict, optional
            optional parameters to be passed to the plot, such as:
                * ax: Axes object to plot on
                * figsize: tuple specifying figure dimensions
                * ylims: list to manually specify y-axis limits, float to specify y-axis margin (see ``plt.Axes.margin()``
                for further information), None to automatically infer y-axis limits

    Returns
    -------

    """

    fig: Union[plt.Figure, None] = None
    ax: Union[plt.Axes, None] = kwargs.get("ax", None)

    if ax is None:
        figsize = kwargs.get("figsize", plt.rcParams["figure.figsize"])
        fig, ax = plt.subplots(figsize=figsize)

    saliva_params = _saliva_params.copy()

    bg_color = kwargs.get(
        "background_color", saliva_params.get("background_color", None)
    )
    bg_alpha = kwargs.get(
        "background_alpha", saliva_params.get("background_alpha", None)
    )
    x_padding = kwargs.get("x_padding", saliva_params.get("x_padding", None))
    ylims = kwargs.get("ylims", None)
    test_text = kwargs.get("test_text", saliva_params["test_text"])
    test_color = kwargs.get("test_color", saliva_params["test_color"])
    test_alpha = kwargs.get("test_alpha", saliva_params["test_alpha"])
    fontsize = kwargs.get("fontsize", saliva_params["fontsize"])
    xaxis_label = kwargs.get("xaxis_label", saliva_params["xaxis_label"])
    xaxis_tick_locator = kwargs.get(
        "xaxis_tick_locator", saliva_params["xaxis_tick_locator"]
    )

    if isinstance(data, dict) and biomarker in data.keys():
        # multiple biomarkers were passed => get the selected biomarker and try to get the groups from the index
        data = data[biomarker]

    if saliva_times is None:
        if isinstance(data, pd.DataFrame):
            # DataFrame was passed
            if "time" in data.index.names:
                saliva_times = np.array(data.index.get_level_values("time").unique())
        else:
            # Dict was passed => multiple groups (where each entry is a dataframe per group) or multiple biomarker
            # (where each entry is one biomarker)
            if all(["time" in d.index.names for d in data.values()]):
                saliva_times = np.array(
                    [d.index.get_level_values("time").unique() for d in data.values()],
                    dtype=object,
                )
                if not all(
                    [
                        len(saliva_time) == len(saliva_times[0])
                        for saliva_time in saliva_times
                    ]
                ):
                    raise ValueError(
                        "Different saliva time lengths passed! Did you pass multiple biomarkers? "
                        "For plotting multiple biomarkers, call the `saliva_plot` function on the same axis "
                        "repeatedly for the different biomarkers!"
                    )
                if (saliva_times == saliva_times[0]).all():
                    saliva_times = saliva_times[0]
                else:
                    raise ValueError(
                        "Saliva times inconsistent for the different groups!"
                    )
            else:
                raise ValueError(
                    "Not all dataframes contain a 'time' column for saliva times!"
                )

    if not groups:
        # extract groups from data if they were not supplied
        if isinstance(data, pd.DataFrame):
            # get group names from index
            if "condition" in data.index.names:
                groups = list(data.index.get_level_values("condition").unique())
            elif group_col:
                if group_col in data:
                    groups = list(data[group_col].unique())
                else:
                    raise ValueError(
                        "`{}`, specified as `group_col` not in columns of the dataframe!".format(
                            group_col
                        )
                    )
            else:
                groups = ["Data"]
        else:
            # get group names from dict
            groups = list(data.keys())

    if saliva_times is None:
        raise ValueError("Must specify saliva times!")

    if len(ax.lines) == 0:
        line_colors = saliva_params["colormap"]
        _saliva_plot_helper(
            data,
            biomarker,
            groups,
            saliva_times,
            ylims=ylims,
            fontsize=fontsize,
            ax=ax,
            line_colors=line_colors,
        )

        ax.text(
            x=test_times[0] + 0.5 * (test_times[1] - test_times[0]),
            y=0.95,
            transform=ax.get_xaxis_transform(),
            s=test_text,
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=fontsize,
        )

        ax.axvspan(*test_times, color=test_color, alpha=test_alpha, zorder=1, lw=0)
        if bg_color is not None:
            if x_padding is None:
                total_length = saliva_times[-1] - saliva_times[0]
                x_padding = 0.1 * total_length
            ax.axvspan(
                saliva_times[0] - x_padding,
                test_times[0],
                color=bg_color,
                alpha=bg_alpha,
                zorder=0,
                lw=0,
            )
            ax.axvspan(
                test_times[1],
                saliva_times[-1] + x_padding,
                color=bg_color,
                alpha=bg_alpha,
                zorder=0,
                lw=0,
            )
            ax.set_xlim(saliva_times[0] - x_padding, saliva_times[-1] + x_padding)

        ax.xaxis.set_major_locator(xaxis_tick_locator)
        ax.set_xlabel(xaxis_label, fontsize=fontsize)
    else:
        # the was already something drawn into the axis => we are using the same axis to add another feature
        ax_twin = ax.twinx()
        line_colors = saliva_params["multi_colormap"]
        _saliva_plot_helper(
            data,
            biomarker,
            groups,
            saliva_times,
            ylims=ylims,
            fontsize=fontsize,
            ax=ax_twin,
            x_offset_basis=saliva_params["multi_x_offset"],
            line_colors=line_colors,
        )

    if len(groups) > 1:
        # get handles
        handles, labels = ax.get_legend_handles_labels()
        # remove the errorbars
        handles = [h[0] for h in handles]
        # use them in the legend
        ax.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),
            numpoints=1,
            prop={"size": fontsize},
        )

    if fig:
        fig.tight_layout()
        return fig, ax


def _saliva_plot_helper(
    data: pd.DataFrame,
    biomarker: str,
    groups: Sequence[str],
    saliva_times: Sequence[int],
    plot_params: Optional[Dict] = None,
    **kwargs
) -> plt.Axes:
    saliva_params = _saliva_params.copy()
    if plot_params:
        saliva_params.update(plot_params)

    ax: Union[plt.Axes, None] = kwargs.get("ax", None)
    # get all plot parameter
    ylims = kwargs.get("ylims", None)
    fontsize = kwargs.get("fontsize", saliva_params["fontsize"])
    line_colors = kwargs.get("line_colors", saliva_params["colormap"])
    x_offset_basis = kwargs.get("x_offset_basis", 0)
    line_styles = saliva_params["line_styles"]
    markers = saliva_params["markers"]
    yaxis_label = saliva_params["yaxis_label"][biomarker]
    x_offsets = list(np.array(saliva_params["x_offsets"]) + x_offset_basis)

    for group, x_off, line_color, marker, ls in zip(
        groups, x_offsets, line_colors, markers, line_styles
    ):
        if group == "Data":
            # no condition index
            df_grp = data
        else:
            df_grp = data.xs(group, level="condition")
        ax.errorbar(
            x=saliva_times + x_off,
            y=df_grp["mean"],
            label=group,
            yerr=df_grp["se"],
            capsize=3,
            marker=marker,
            color=line_color,
            ls=ls,
        )

    ax.set_ylabel(yaxis_label, fontsize=fontsize)

    if isinstance(ylims, (tuple, list)):
        ax.set_ylim(ylims)
    else:
        ymargin = 0.15
        if isinstance(ylims, float):
            ymargin = ylims
        ax.margins(x=0.05, y=ymargin)

    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    return ax


def saliva_plot_combine_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    biomarkers: Sequence[str],
    separate_legends: Optional[bool] = False,
    plot_params: Optional[Dict] = None,
):
    """
    TODO: add documentation

    Parameters
    ----------
    fig
    ax
    biomarkers
    separate_legends
    plot_params

    Returns
    -------

    """
    from matplotlib.legend_handler import HandlerTuple

    saliva_params = _saliva_params.copy()
    if plot_params:
        saliva_params.update(plot_params)

    fontsize = saliva_params["multi_fontsize"]
    legend_offset = saliva_params["multi_legend_offset"]

    labels = [ax.get_legend_handles_labels()[1] for ax in fig.get_axes()]
    if all([len(l) == 1 for l in labels]):
        # only one group
        handles = [ax.get_legend_handles_labels()[0] for ax in fig.get_axes()]
        handles = [h[0] for handle in handles for h in handle]
        labels = biomarkers
        ax.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),
            prop={"size": fontsize},
        )
    else:
        if separate_legends:
            for (i, a), biomarker in zip(
                enumerate(reversed(fig.get_axes())), reversed(biomarkers)
            ):
                handles, labels = a.get_legend_handles_labels()
                l = ax.legend(
                    handles,
                    labels,
                    title=biomarker,
                    loc="upper right",
                    bbox_to_anchor=(0.99 - legend_offset * i, 0.99),
                    prop={"size": fontsize},
                )
                ax.add_artist(l)
        else:
            handles = [ax.get_legend_handles_labels()[0] for ax in fig.get_axes()]
            handles = [h[0] for handle in handles for h in handle]
            labels = [ax.get_legend_handles_labels()[1] for ax in fig.get_axes()]
            labels = [
                "{}:\n{}".format(b, " - ".join(l)) for b, l in zip(biomarkers, labels)
            ]
            ax.legend(
                list(zip(handles[::2], handles[1::2])),
                labels,
                loc="upper right",
                bbox_to_anchor=(0.99, 0.99),
                numpoints=1,
                handler_map={tuple: HandlerTuple(ndivide=None)},
                prop={"size": fontsize},
            )


def saliva_feature_boxplot(
    data: pd.DataFrame,
    x: str,
    saliva_type: str,
    feature: Optional[str] = None,
    stats_kwargs: Optional[Dict] = None,
    **kwargs
):
    from biopsykit.plotting import feature_boxplot

    if feature is not None:
        if isinstance(feature, str):
            feature = [feature]
        ylabel = _get_ylabels(saliva_type, feature)
        ylabel = [ylabel[f] for f in feature]
        if len(set(ylabel)) == 1:
            kwargs["ylabel"] = ylabel[0]
    return feature_boxplot(
        data=data, x=x, y=saliva_type, stats_kwargs=stats_kwargs, **kwargs
    )


def saliva_multi_feature_boxplot(
    data: pd.DataFrame,
    saliva_type: str,
    features: Union[str, Sequence[str]],
    hue: Optional[str] = None,
    xticklabels: Optional[Dict[str, str]] = None,
    stats_kwargs: Optional[Dict] = None,
    **kwargs
):
    """

    Parameters
    ----------
    data
    saliva_type
    hue
    features
    filter_features : bool, optional
        ``True`` to filter features by name, ``False`` to match exact feature names. Default: ``True``
    xticklabels
    stats_kwargs
    kwargs

    Returns
    -------

    """
    from biopsykit.plotting import multi_feature_boxplot

    x = kwargs.get("x", "biomarker")

    if isinstance(features, str):
        # ensure list
        features = [features]

    if xticklabels is None:
        xticklabels = _get_xticklabels(data, features)

    if "ylabels" not in kwargs:
        kwargs["ylabels"] = _get_ylabels(saliva_type, features)

    return multi_feature_boxplot(
        data,
        x=x,
        y=saliva_type,
        hue=hue,
        features=features,
        xticklabels=xticklabels,
        stats_kwargs=stats_kwargs,
        **kwargs
    )


def _get_xticklabels(
    data: pd.DataFrame, features: Sequence[str]
) -> Dict[str, Sequence[str]]:
    import re

    xlabel_dict = {}
    for feature in features:
        cols = features[feature]
        if isinstance(cols, str):
            cols = [cols]
        labels = []
        for c in cols:
            if "slope" in c:
                label = _saliva_feature_params["xticklabels"]["slope"].replace(
                    "§", re.findall(r"slope(\w+)", c)[0]
                )
            else:
                label = _saliva_feature_params["xticklabels"][c]
            labels.append(label)

        xlabel_dict[feature] = labels
    return xlabel_dict


def _get_ylabels(saliva_type: str, features: Union[str, Sequence[str]]):
    ylabels = _saliva_feature_params["yaxis_label"][saliva_type]
    if isinstance(features, str):
        features = [features]
    for feature in features:
        if "slope" in feature:
            ylabels[feature] = ylabels["slope"]

    return ylabels
