from typing import Union, Tuple, Sequence, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from biopsykit.utils.functions import se


def lineplot(data: pd.DataFrame, **kwargs) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
    markers = None
    style = kwargs.get('style', None)
    ax: plt.Axes = kwargs.get('ax', None)
    hue = kwargs.get('hue')
    x = kwargs.get('x')
    y = kwargs.get('y')
    order = kwargs.get('order')

    if style is not None:
        markers = ['o'] * len(data.index.get_level_values(style).unique())

    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    kwargs.update(
        {'dashes': False, 'err_style': 'bars', 'markers': markers, 'ci': 68, 'ax': ax, 'err_kws': {'capsize': 5}}
    )

    data = data.reset_index()
    grouped = {key: val for key, val in data.groupby(hue)}

    if order is not None:
        # reorder group dictionary
        grouped = {key: grouped[key] for key in order}

    x_vals = np.arange(0, len(data[x].unique()))
    for i, (key, df) in enumerate(grouped.items()):
        m_se = df.groupby([x, hue]).agg([np.mean, se])[y]
        err_kws = kwargs.get('err_kws')
        marker = markers[i] if markers else None
        ax.errorbar(x=x_vals + 0.05 * i, y=m_se['mean'], yerr=m_se['se'].values, marker=marker, label=key, **err_kws)

    ylabel = kwargs.get('ylabel', data[y].name)
    xlabel = kwargs.get('xlabel', data[x].name)
    xticklabels = kwargs.get('xticklabels', data[x].unique())
    ylim = kwargs.get('ylim', None)

    ax.set_xticks(x_vals)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)

    # get handles
    handles, labels = ax.get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    # use them in the legend
    ax.legend(handles, labels, loc='best', numpoints=1)

    if fig is not None:
        return fig, ax


def stacked_barchart(data: pd.DataFrame, **kwargs) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
    fig = None
    ax: plt.Axes = kwargs.get('ax', None)
    if ax is None:
        fig, ax = plt.subplots()

    ylabel = kwargs.get('ylabel', None)
    order = kwargs.get('order', None)
    if order:
        data = data.reindex(order)

    ax = data.plot(kind='bar', stacked=True, ax=ax, rot=0)
    ax.legend().set_title(None)
    if ylabel:
        ax.set_ylabel(ylabel)

    return fig, ax


def multi_feature_boxplot(data: pd.DataFrame, x: str, y: str, hue: str,
                          features: Sequence[str], filter_features: Optional[bool] = True,
                          xticklabels: Optional[Dict[str, Sequence[str]]] = None,
                          ylabels: Optional[Dict[str, str]] = None,
                          stats_kwargs: Optional[Dict] = None,
                          **kwargs) -> Union[None, Tuple[plt.Figure, Sequence[plt.Axes]]]:
    """

    Parameters
    ----------
    data
    x
    y
    hue
    features
    filter_features : bool, optional
        ``True`` to filter features by name, ``False`` to match exact feature names. Default: ``True``
    xticklabels
    ylabels
    stats_kwargs
    kwargs

    Returns
    -------

    """
    from statannot import add_stat_annotation

    axs: Sequence[plt.Axes] = kwargs.get('axs', kwargs.get('ax', None))
    hue_order = kwargs.get('hue_order', None)
    notch = kwargs.get('notch', False)
    legend = kwargs.get('legend', True)
    xlabels = kwargs.get('xlabels', {})
    rect = kwargs.get('rect', (0, 0, 0.825, 1.0))
    pvalue_thresholds = [[1e-3, "***"], [1e-2, "**"], [0.05, "*"]]

    if ylabels is None:
        ylabels = {}
    if xticklabels is None:
        xticklabels = {}

    if axs is None:
        fig, axs = plt.subplots(figsize=(15, 5), ncols=len(features))
    else:
        fig = axs[0].get_figure()

    boxplot_pairs = {}
    if stats_kwargs is not None:
        boxplot_pairs = stats_kwargs.pop('boxplot_pairs')

    h, l = None, None
    for ax, feature in zip(axs, features):
        if filter_features:
            data_plot = data.unstack().filter(like=feature).stack()
        else:
            data_plot = data.unstack().loc[:, pd.IndexSlice[:, feature]].stack()
        sns.boxplot(data=data_plot.reset_index(), x=x, y=y, hue=hue, hue_order=hue_order, ax=ax, notch=notch)

        if stats_kwargs is not None:
            box_pairs = boxplot_pairs.get(feature, [])
            stats_kwargs['comparisons_correction'] = stats_kwargs.get('comparisons_correction', None)
            stats_kwargs['test'] = stats_kwargs.get('test', 't-test_ind')
            stats_kwargs['pvalue_thresholds'] = pvalue_thresholds

            if len(box_pairs) > 0:
                add_stat_annotation(ax=ax, data=data_plot.reset_index(), box_pairs=box_pairs, x=x, y=y,
                                    hue=hue, hue_order=hue_order, **stats_kwargs)

        if feature in ylabels:
            ax.set_ylabel(ylabels[feature])

        if feature in xticklabels:
            if feature in xlabels:
                ax.set_xlabel(xlabels[feature])
            else:
                ax.set_xlabel(None)
            ax.set_xticklabels(xticklabels[feature])
        h, l = ax.get_legend_handles_labels()
        ax.legend().remove()

    if fig is not None:
        if legend:
            fig.legend(h, l, loc='upper right', bbox_to_anchor=(1.0, 1.0))
            fig.tight_layout(pad=0.5, rect=rect)
        return fig, axs
