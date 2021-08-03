"""Module containing several advanced plotting functions."""
from typing import Union, Tuple, Sequence, Optional, Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statannot import add_stat_annotation


from biopsykit.utils.dataframe_handling import multi_xs
from biopsykit.utils.functions import se

_PVALUE_THRESHOLDS = [[1e-3, "***"], [1e-2, "**"], [0.05, "*"]]


def lineplot(
    data: pd.DataFrame, x: str, y: str, hue: Optional[str] = None, style: Optional[str] = None, **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Draw a line plot with error bars with the possibility of several semantic groupings.

    This is an extension to seaborn's lineplot function (:func:`seaborn.lineplot`).
    It offers the same interface, but several improvements:

    * Data points are not only connected as line, but are also drawn with marker.
    * Lines can have an offset along the categorical (x) axis for better visualization
      (seaborn equivalent: ``dodge``, which is only available for :func:`seaborn.pointplot`,
      not for :func:`seaborn.lineplot`).
    * Further plot parameters (axis labels, ticks, etc.) are inferred from the dataframe.

    Equivalent to seaborn, the relationship between ``x`` and ``y`` can be shown for different subsets of the
    data using the ``hue`` and ``style`` parameters. If both parameters are assigned two different grouping variables
    can be represented.

    Error bars are displayed as standard error.

    See the seaborn documentation for further information.


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to plot
    x : str
        column of x axis in ``data``
    y : str
        column of y axis in ``data``
    hue : str, optional
        column name of grouping variable that will produce lines with different colors.
        Can be either categorical or numeric. If ``None`` then data will not be grouped.
    style : str, optional
        column name of grouping variable that will produce lines with different dashes and/or marker.
        If ``None`` then lines will not have different styles.
    **kwargs
        Additional parameters to configure the plot. Parameters include:

        * ``x_offset``: offset value to move different groups along the x axis for better visualization.
          Default: 0.05.
        * ``xlabel``: Label for x axis. If not specified it is inferred from the ``x`` column name.
        * ``ylabel``: Label for y axis. If not specified it is inferred from the ``y`` column name.
        * ``xticklabels``: List of labels for ticks of x axis. If not specified ``order`` is taken as tick labels.
          If ``order`` is not specified tick labels are inferred from x values.
        * ``ylim``: y-axis limits.
        * ``order``: list specifying the order of categorical values along the x axis.
        * ``hue_order``: list specifying the order of processing and plotting for categorical levels
          of the ``hue`` semantic.
        * ``marker``: string or list of strings to specify marker style.
          If ``marker`` is a string, then marker of each line will have the same style.
          If ``marker`` is a list, then marker of each line will have a different style.
        * ``linestyle``: string or list of strings to specify line style.
          If ``linestyle`` is a string, then each line will have the same style.
          If ``linestyle`` is a list, then each line will have a different style.
        * ``legend_fontsize``: font size of legend.
        * ``legend_loc``: location of legend in Axes.
        * ``ax``: pre-existing axes for the plot. Otherwise, a new figure and axes object is created and returned.
        * ``err_kws``: additional parameters to control the aesthetics of the error bars.
          The ``err_kws`` are passed down to :meth:`matplotlib.axes.Axes.errorbar` or
          :meth:`matplotlib.axes.Axes.fill_between`, depending on ``err_style``. Parameters include:

          * ``capsize``: length of error bar caps in points


    Returns
    -------
    fig : :class:`~matplotlib.figure.Figure`
        figure object
    ax : :class:`~matplotlib.axes.Axes`
        axes object


    See Also
    --------
    :func:`seaborn.lineplot`
        line plot function of Seaborn

    """
    x_offset = kwargs.get("x_offset", 0.01)
    multi_x_offset = kwargs.get("multi_x_offset", 0)
    order = kwargs.get("order")
    marker = kwargs.get("marker", None)
    linestyle = kwargs.get("linestyle", None)
    hue_order = kwargs.get("hue_order")
    colormap = kwargs.get("colormap")
    show_legend = kwargs.get("show_legend", True)

    data = data.reset_index()

    fig, ax = _plot_get_fig_ax(**kwargs)

    kwargs.setdefault("ax", ax)
    kwargs.setdefault("err_kws", {"capsize": 5})

    marker, linestyle = _get_styles(data, style, hue, marker, linestyle)

    if hue is not None:
        grouped = {key: val for key, val in data.groupby(hue)}  # pylint:disable=unnecessary-comprehension
    else:
        grouped = {y: data}

    if hue_order is not None:
        # reorder group dictionary
        grouped = {key: grouped[key] for key in hue_order}

    # generate x axis
    x_vals = data[x].unique()
    if all(isinstance(x, str) for x in x_vals):
        x_vals = np.arange(0, len(x_vals))
    span = x_vals[-1] - x_vals[0]
    x_offset = span * x_offset
    # iterate through groups
    for i, (key, df) in enumerate(grouped.items()):
        m_se = _get_df_lineplot(df, x, y, hue, order)

        err_kws = kwargs.get("err_kws", {})
        m = marker[i] if marker is not None else None
        ls = linestyle[i] if linestyle is not None else "-"
        c = colormap[i] if colormap is not None else None

        ax.errorbar(
            x=x_vals + multi_x_offset * span + x_offset * i,
            y=m_se["mean"],
            yerr=m_se["se"].values,
            marker=m,
            linestyle=ls,
            color=c,
            label=key,
            **err_kws,
        )

    _lineplot_style_axis(data, x, y, x_vals, order, **kwargs)

    if show_legend:
        _set_legend_errorbar(**kwargs)

    fig.tight_layout()
    return fig, ax


def _lineplot_style_axis(data: pd.DataFrame, x: str, y: str, x_vals: np.ndarray, order: str, ax: plt.Axes, **kwargs):
    xlabel = kwargs.get("xlabel", x)
    ylabel = kwargs.get("ylabel", y)
    ylim = kwargs.get("ylim", None)

    if order is None:
        xticklabels = kwargs.get("xticklabels", data.groupby([x]).groups.keys())
    else:
        xticklabels = kwargs.get("xticklabels", order)

    ax.set_xticks(x_vals)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)


def stacked_barchart(data: pd.DataFrame, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Draw a stacked bar chart.

    A stacked bar chart has multiple bar charts along a categorical axis (``x`` axis) where values are
    stacked along the value axis (``y`` axis). The categorical axis corresponds to the columns in the dataframe
    whereas the value axis corresponds to the rows.

    This is an extension to the already existing function provided by pandas
    (``pandas.DataFrame.plot(kind='bar', stacked=True)``).


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to plot
    **kwargs
        Additional parameters to plotting function. For example, this can be:

        * ``order``: order of items along the categorical axis.
        * ``ylabel``: label of y axis.
        * ``ax``: pre-existing axes for the plot. Otherwise, a new figure and axes object is created and returned.


    Returns
    -------
    fig : :class:`~matplotlib.figure.Figure`
        figure object
    ax : :class:`~matplotlib.axes.Axes`
        axes object


    See Also
    --------
    :func:`~biopsykit.utils.dataframe_handling.stack_groups_percent`
        function to rearrange dataframe to be plotted as stacked bar chart

    """
    ax: plt.Axes = kwargs.get("ax", None)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ylabel = kwargs.get("ylabel", None)
    order = kwargs.get("order", None)
    if order:
        data = data.reindex(order)

    ax = data.plot(kind="bar", stacked=True, ax=ax, rot=0)
    ax.legend().set_title(None)
    if ylabel:
        ax.set_ylabel(ylabel)

    fig.tight_layout()
    return fig, ax


def feature_boxplot(
    data: pd.DataFrame,
    x: Optional[str] = None,
    y: Optional[str] = None,
    order: Optional[Sequence[str]] = None,
    hue: Optional[str] = None,
    hue_order: Optional[Sequence[str]] = None,
    stats_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Draw boxplot with significance brackets.

    This is a wrapper of seaborn's boxplot function (:func:`~seaborn.boxplot`) that allows to add significance
    brackets that indicate statistical significance.


    Statistical annotations are plotted using ``statannot`` (https://github.com/webermarcolivier/statannot).
    This library can either use existing statistical results or perform statistical tests internally.
    To plot significance brackets a list of box pairs where annotations should be added are required.
    The p values can be provided as well, or, alternatively, be computed by ``statannot``.
    If :class:`~biopsykit.stats.StatsPipeline` was used for statistical analysis the list of box pairs and p values
    can be generated using :func:`~biopsykit.stats.StatsPipeline.sig_brackets` and passed in the ``stats_kws``
    parameter. Otherwise, see the ``statannot`` documentation  for a tutorial on how to specify significance brackets.

    .. note::
        The input data is assumed to be in long-format.


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to plot
    x : str
        column of x axis in ``data``
    y : str
        column of y axis in ``data``
    order : list str, optional
        order to plot the categorical levels along the x axis.
    hue : str, optional
        column name of grouping variable. Default: ``None``
    hue_order : list of str, optional
        order to plot the grouping variable specified by ``hue``
    stats_kwargs : dict, optional
        dictionary with arguments for significance brackets.

        If annotations should be added, the following parameter is required:

        * ``box_pairs``: list of box pairs that should be annotated

        If already existing box pairs and p values should be used the following parameter is additionally required:

        * ``pvalues``: list of p values corresponding to ``box_pairs``

        If statistical tests should be computed by ``statsannot``, the following parameters are required:

        * ``test``: type of statistical test to be computed
        * ``comparisons_correction`` (optional): Whether (and which) type of multi-comparison correction should be
          applied. ``None`` to not apply any multi-comparison (default).

        The following parameters are optional:

        * ``pvalue_thresholds``: list of p value thresholds for statistical annotations. The default annotation is:
          '*': 0.01 <= p < 0.05, '**': 0.001 <= p < 0.01, `'***'`: p < 0.001
          (:math:`[[1e-3, "***"], [1e-2, "**"], [0.05, "*"]]`)

    **kwargs
        additional arguments that are passed down to :func:`~seaborn.boxplot`, for example:

        * ``ylabel``: label of y axis
        * ``ax``: pre-existing axes for the plot. Otherwise, a new figure and axes object is created and returned.


    Returns
    -------
    fig : :class:`~matplotlib.figure.Figure`
        figure object
    ax : :class:`~matplotlib.axes.Axes`
        axes object


    See Also
    --------
    :func:`~seaborn.boxplot`
        seaborn function to create boxplots
    :class:`~biopsykit.stats.StatsPipeline`
        class to create statistical analysis pipelines

    """
    ax: plt.Axes = kwargs.pop("ax", None)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if stats_kwargs is None:
        stats_kwargs = {}

    ylabel = kwargs.pop("ylabel", None)

    stats_kwargs = _feature_boxplot_sanitize_stats_kwargs(stats_kwargs)

    box_pairs = stats_kwargs.get("box_pairs", {})
    if len(box_pairs) == 0:
        stats_kwargs = {}

    sns.boxplot(data=data.reset_index(), x=x, y=y, ax=ax, order=order, hue=hue, hue_order=hue_order, **kwargs)
    if len(box_pairs) > 0:
        add_stat_annotation(
            data=data.reset_index(), ax=ax, x=x, y=y, order=order, hue=hue, hue_order=hue_order, **stats_kwargs
        )

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    fig.tight_layout()
    return fig, ax


def _feature_boxplot_sanitize_stats_kwargs(stats_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    boxplot_pvals = stats_kwargs.get("pvalues", {})
    if len(stats_kwargs) > 0:
        stats_kwargs["comparisons_correction"] = stats_kwargs.get("comparisons_correction", None)
        stats_kwargs["test"] = stats_kwargs.get("test", None)

    if len(boxplot_pvals) > 0:
        stats_kwargs["perform_stat_test"] = False

    stats_kwargs.setdefault("pvalue_thresholds", _PVALUE_THRESHOLDS)
    return stats_kwargs


# TODO "group" parameter should always be "x"? check if "group" can be omitted
def multi_feature_boxplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    features: Union[Sequence[str], Dict[str, Union[str, Sequence[str]]]],
    group: str,
    order: Optional[Sequence[str]] = None,
    hue: Optional[str] = None,
    hue_order: Optional[Sequence[str]] = None,
    stats_kwargs: Optional[Dict] = None,
    **kwargs,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Draw multiple features as boxplots with significance brackets.

    For each feature, a new subplot will be created. Similarly to `feature_boxplot` subplots can be annotated
    with statistical significance brackets (can be specified via ``stats_kwargs`` parameter). For further information,
    see the :func:`~biopsykit.plotting.feature_boxplot` documentation.

    .. note::
        The input data is assumed to be in *long*-format.


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to plot
    x : str
        column of x axis in ``data``
    y : str
        column of y axis in ``data``
    features : list of str or dict of str
        features to plot. If ``features`` is a list, each entry must correspond to one feature category in the
        index level specified by ``group``. A separate subplot will be created for each feature.
        If similar features (i.e., different `slope` or `AUC` parameters) should be combined into one subplot,
        ``features`` can be provided as dictionary.
        Then, the dict keys specify the feature category (a separate subplot will be created for each category)
        and the dict values specify the feature (or list of features) that are combined into the subplots.
    group : str
        name of index level with feature names. Corresponds to the subplots that are created.
    order : list of str, optional
        order to plot the categorical levels along the x axis
    hue : str, optional
        column name of grouping variable. Default: ``None``
    hue_order : list of str, optional
        order to plot the grouping variable specified by ``hue``
    stats_kwargs : dict, optional
        nested dictionary with arguments for significance brackets. The behavior and expected parameters are similar
        to the ``stats_kwargs`` parameter in :func:`~biopsykit.plotting.feature_boxplot`. However, the ``box_pairs``
        and ``pvalues`` arguments are expected not to be lists, but dictionaries with keys corresponding to the list
        entries (or the dict keys) in ``features`` and box pair / p-value lists are the dict values.
    **kwargs
        additional arguments that are passed down to :func:`~seaborn.boxplot`. For example:

        * ``order``: specifies x axis order for subplots. Can be a list if order is the same for all subplots or a
          dict if order should be individual for subplots
        * ``xticklabels``: dictionary to set tick labels of x axis in subplots. Keys correspond to the list entries
          (or the dict keys) in ``features``. Default: ``None``
        * ``ylabels``: dictionary to set y axis labels in subplots. Keys correspond to the list entries
          (or the dict keys) in ``features``. Default: ``None``
        * ``axs``: list of pre-existing axes for the plot. Otherwise, a new figure and axes object is created and
          returned.


    Returns
    -------
    fig : :class:`~matplotlib.figure.Figure`
        figure object
    axs : list of :class:`~matplotlib.axes.Axes`
        list of subplot axes objects


    See Also
    --------
    :func:`~seaborn.boxplot`
        seaborn function to create boxplots
    :class:`~biopsykit.plotting.feature_boxplot`
        plot single feature boxplot
    :class:`~biopsykit.stats.StatsPipeline`
        class to create statistical analysis pipelines and get parameter for plotting significance brackets

    """
    xlabels = kwargs.pop("xlabels", {})
    ylabels = kwargs.pop("ylabels", {})
    xticklabels = kwargs.pop("xticklabels", {})
    show_legend = kwargs.pop("show_legend", True)
    rect = kwargs.pop("rect", (0, 0, 0.825, 1.0))
    legend_fontsize = kwargs.pop("legend_fontsize", None)
    legend_loc = kwargs.pop("legend_loc", "upper right")
    legend_orientation = kwargs.pop("legend_orientation", "vertical")

    if isinstance(features, list):
        features = {f: f for f in features}

    fig, axs = _plot_get_fig_ax_list(features, **kwargs)

    if stats_kwargs is None:
        stats_kwargs = {}

    dict_box_pairs = stats_kwargs.pop("box_pairs", None)
    dict_pvals = stats_kwargs.pop("pvalues", None)

    handles = None
    labels = None
    for ax, key in zip(axs, features):
        data_plot = multi_xs(data, features[key], level=group)
        if data_plot.empty:
            raise ValueError("Empty dataframe for '{}'!".format(key))

        order_list = _multi_feature_boxplot_get_order(order, key)

        sns.boxplot(
            data=data_plot.reset_index(), x=x, y=y, order=order_list, hue=hue, hue_order=hue_order, ax=ax, **kwargs
        )

        _add_stat_annot_multi_feature_boxplot(
            data_plot, x, y, order_list, hue, hue_order, stats_kwargs, dict_box_pairs, dict_pvals, key, ax
        )

        ax.set_ylabel(ylabels.get(key, ax.get_ylabel()))
        _style_xaxis_multi_feature_boxplot(ax, xlabels, xticklabels, key)

        if hue is not None:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend().remove()

    if show_legend:
        _add_legend_multi_feature_boxplot(
            fig,
            hue,
            handles,
            labels,
            rect=rect,
            legend_loc=legend_loc,
            legend_fontsize=legend_fontsize,
            legend_orientation=legend_orientation,
        )

    fig.tight_layout(rect=rect)
    return fig, axs


def _multi_feature_boxplot_get_order(order: Union[Dict[str, Sequence[str]], Sequence[str]], key: str):
    if isinstance(order, dict):
        return order[key]
    return order


def _get_df_lineplot(data: pd.DataFrame, x: str, y: str, hue: str, order: Sequence[str]) -> pd.DataFrame:
    if "mean" in data.columns and "se" in data.columns:
        m_se = data
    else:
        if hue is None:
            m_se = data.groupby([x]).agg([np.mean, se])[y]
        else:
            m_se = data.groupby([x, hue]).agg([np.mean, se])[y]
    if order is not None:
        m_se = m_se.reindex(order, level=0)
    return m_se


def _set_legend_errorbar(**kwargs):
    legend_fontsize = kwargs.get("legend_fontsize", "small")
    legend_loc = kwargs.get("legend_loc", "upper left")
    ax = kwargs.get("ax")

    # get handles
    handles, labels = ax.get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    # use them in the legend
    ax.legend(handles, labels, loc=legend_loc, numpoints=1, fontsize=legend_fontsize)


def _get_styles(
    data: pd.DataFrame,
    style: str,
    hue: str,
    marker: Optional[Union[str, Sequence[str]]] = None,
    linestyle: Optional[Union[str, Sequence[str]]] = None,
) -> Tuple[Sequence[str], Sequence[str]]:
    num_cats = None
    if style is None:
        if all(v is not None for v in [hue, marker, linestyle]):
            num_cats = len(data[hue].unique())
            marker, linestyle = _get_marker_linestyle(marker, linestyle, num_cats)
        else:
            marker, linestyle = _get_marker_linestyle(marker, linestyle, 1)
    else:
        num_cats = len(data[style].unique())
        marker, linestyle = _get_marker_linestyle_style(marker, linestyle, num_cats)

    _get_styles_assert_styles(marker, linestyle, num_cats)

    return marker, linestyle


def _get_marker_linestyle(
    marker: Union[str, Sequence[str]], linestyle: Union[str, Sequence[str]], num_cats: int
) -> Tuple[Sequence[str], Sequence[str]]:
    if isinstance(marker, str):
        marker = [marker] * num_cats
    if isinstance(linestyle, str):
        linestyle = [linestyle] * num_cats
    return marker, linestyle


def _get_marker_linestyle_style(
    marker: Union[str, Sequence[str]], linestyle: Union[str, Sequence[str]], num_cats: int
) -> Tuple[Sequence[str], Sequence[str]]:
    if marker is None:
        marker = "o"
    if linestyle is None:
        linestyle = "-"
    return _get_marker_linestyle(marker, linestyle, num_cats)


def _get_styles_assert_styles(marker: Sequence[str], linestyle: Sequence[str], num_cats: int):
    if num_cats is not None:
        if len(marker) != num_cats:
            raise ValueError("If a list of 'marker' is provided it must match the number of 'style' categories!")
        if len(linestyle) != num_cats:
            raise ValueError("If a list of 'linestyle' is provided it must match the number of 'style' categories!")


def _add_stat_annot_multi_feature_boxplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    order: Sequence[str],
    hue: str,
    hue_order: Sequence[str],
    stats_kwargs: Dict,
    dict_box_pairs: Dict,
    dict_pvals: Dict,
    key: str,
    ax: plt.Axes,
):
    if len(stats_kwargs) > 0:
        stats_kwargs["comparisons_correction"] = stats_kwargs.get("comparisons_correction", None)
        stats_kwargs["test"] = stats_kwargs.get("test", None)

    if dict_box_pairs is not None:
        # filter box pairs by feature
        stats_kwargs["box_pairs"] = [dict_box_pairs[x] for x in dict_box_pairs if key in x]
        # flatten list
        stats_kwargs["box_pairs"] = [x for pairs in stats_kwargs["box_pairs"] for x in pairs]

    if dict_pvals is not None:
        # filter pvals by feature
        stats_kwargs["pvalues"] = [dict_pvals[x] for x in dict_pvals if key in x]
        # flatten list
        stats_kwargs["pvalues"] = [x for pairs in stats_kwargs["pvalues"] for x in pairs]
        stats_kwargs["perform_stat_test"] = False

    stats_kwargs["pvalue_thresholds"] = _PVALUE_THRESHOLDS

    if "box_pairs" in stats_kwargs and len(stats_kwargs["box_pairs"]) > 0:
        add_stat_annotation(
            ax=ax,
            data=data.reset_index(),
            x=x,
            y=y,
            order=order,
            hue=hue,
            hue_order=hue_order,
            **stats_kwargs,
        )


def _add_legend_multi_feature_boxplot(fig: plt.Figure, hue: str, handles: Sequence, labels: Sequence, **kwargs):
    legend_fontsize = kwargs.get("legend_fontsize")
    legend_loc = kwargs.get("legend_loc")
    legend_orientation = kwargs.get("legend_orientation")
    rect = kwargs.get("rect")

    if hue is not None:
        ncol = len(handles) if legend_orientation == "horizontal" else 1

        fig.legend(
            handles,
            labels,
            loc=legend_loc,
            ncol=ncol,
            fontsize=legend_fontsize,
        )
    fig.tight_layout(pad=0.5, rect=rect)


def _style_xaxis_multi_feature_boxplot(
    ax: plt.Axes, xlabels: Dict[str, str], xticklabels: Dict[str, Union[str, Sequence[str]]], key: str
):
    if key in xticklabels:
        ax.set_xlabel(xlabels.get(key, None))
        xt = xticklabels[key]
        if isinstance(xt, str):
            xt = [xt]
        ax.set_xticklabels(xt)


def _plot_get_fig_ax(**kwargs):
    ax: plt.Axes = kwargs.get("ax", None)
    if ax is None:
        figsize = kwargs.get("figsize")
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


def _plot_get_fig_ax_list(
    features: Dict[str, Union[str, Sequence[str]]], **kwargs
) -> Tuple[plt.Figure, List[plt.Axes]]:
    axs: List[plt.Axes] = kwargs.pop("axs", kwargs.pop("ax", None))
    figsize = kwargs.pop("figsize", (15, 5))

    if axs is None:
        fig, axs = plt.subplots(figsize=figsize, ncols=len(features))
    else:
        fig = axs[0].get_figure()
    return fig, axs
