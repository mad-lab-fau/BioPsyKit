from typing import Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    if fig is not None:
        return fig, ax
