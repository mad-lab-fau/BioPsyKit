from typing import Union, Optional, Dict, Tuple

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.transforms as mtrans

import biopsykit.colors as colors

sleep_imu_plot_params = {
    'background.color': ['#e0e0e0', '#9e9e9e'],
    'background.alpha': [0.3, 0.3],
}


def sleep_imu_plot(data: pd.DataFrame,
                   sleep_endpoints: Union[Dict, pd.DataFrame],
                   downsample_factor: Optional[float] = 1,
                   ax: Optional[plt.Axes] = None,
                   figsize: Optional[Tuple[float, float]] = None) -> Union[Tuple[plt.Figure, plt.Axes], None]:
    import matplotlib.ticker as mticks

    fig: Union[plt.Figure, None] = None
    sns.set_palette(colors.cmap_fau_blue('3'))

    if ax is None:
        if figsize is None:
            figsize = plt.rcParams['figure.figsize']
        fig, ax = plt.subplots(figsize=figsize)

    if isinstance(data.index, pd.DatetimeIndex):
        plt.rcParams['timezone'] = data.index.tz.zone

    if downsample_factor < 1:
        raise ValueError("Invalid downsample factor!")
    data = data.filter(like="acc")[::downsample_factor]

    mrp_start = pd.to_datetime(sleep_endpoints['major_rest_period_start'])
    mrp_end = pd.to_datetime(sleep_endpoints['major_rest_period_end'])
    sleep_onset = pd.to_datetime(sleep_endpoints['sleep_onset'])
    wake_onset = pd.to_datetime(sleep_endpoints['wake_onset'])

    if isinstance(sleep_endpoints, dict):
        sleep_bouts = sleep_endpoints['sleep_bouts']
        wake_bouts = sleep_endpoints['wake_bouts']
        date = sleep_endpoints['date']
    else:
        sleep_bouts = pd.DataFrame(sleep_endpoints['sleep_bouts'][0])
        wake_bouts = pd.DataFrame(sleep_endpoints['wake_bouts'][0])
        date = sleep_endpoints.index[0][1]

    date = pd.to_datetime(date)

    data.plot(ax=ax)

    # 00:00 (12 am) vline (if present)
    if date == mrp_start.normalize():
        ax.vlines([date + pd.Timedelta("1d")], 0, 1, transform=ax.get_xaxis_transform(),
                  linewidths=3, linestyles='dotted', colors=colors.fau_color('tech'), zorder=0)

    bbox = dict(
        fc=(1, 1, 1, plt.rcParams['legend.framealpha']),
        ec=plt.rcParams['legend.edgecolor'],
        boxstyle="round",
    )

    _plot_sleep_onset(sleep_onset, bbox, ax)
    _plot_wake_onset(wake_onset, bbox, ax)
    _plot_mrp_start(sleep_onset, mrp_start, bbox, ax)
    _plot_mrp_end(wake_onset, mrp_end, bbox, ax)

    handles = _plot_sleep_wake_bouts(sleep_bouts, wake_bouts, ax)

    # wear_time['end'] = wear_time.index.shift(1, freq=pd.Timedelta("15M"))
    # wear_time = wear_time[wear_time['wear'] == 0.0]
    # wear_time = wear_time.reset_index()
    #
    # handle = None
    # for idx, row in wear_time.iterrows():
    #     handle = ax.axvspan(row['index'], row['end'], color=colors.fau_color('wiso'), alpha=0.5, lw=0)
    # if handle is not None:
    #     handles['non-wear'] = handle

    legend = ax.legend(handles=list(handles.values()), labels=list(handles.keys()), loc='lower right', framealpha=1.0,
                       fontsize=14)
    ax.add_artist(legend)

    ax.legend(loc='lower left', framealpha=1.0, fontsize=14)

    ax.set_ylabel("Acceleration [g]")
    ax.set_title("Sleep IMU Data: {} – {}".format(date.date(), (date + pd.Timedelta("1d")).date()))

    if isinstance(data.index, pd.DatetimeIndex):
        # TODO add axis style for non-Datetime axes
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_minor_locator(mticks.AutoMinorLocator(6))

    if fig:
        ax.set_xlabel("Time")
        fig.tight_layout()
        fig.autofmt_xdate(rotation=0, ha='center')
        return fig, ax


def _plot_sleep_onset(sleep_onset, bbox: Dict, ax: plt.Axes):
    # Sleep Onset vline
    ax.vlines([sleep_onset], 0, 1, transform=ax.get_xaxis_transform(), linewidth=3, linestyles='--',
              colors=colors.fau_color('nat'), zorder=3)

    # Sleep Onset Text + Arrow
    ax.annotate(
        "Sleep Onset",
        xy=(mdates.date2num(sleep_onset), 0.90),
        xycoords=mtrans.blended_transform_factory(ax.transData, ax.transAxes),
        xytext=(mdates.date2num(sleep_onset + pd.Timedelta("20M")), 0.90),
        textcoords=mtrans.blended_transform_factory(ax.transData, ax.transAxes),
        ha='left',
        va='center',
        bbox=bbox,
        size=14,
        arrowprops=dict(
            arrowstyle="->",
            lw=2,
            color=colors.fau_color('nat'),
            shrinkA=0.0, shrinkB=0.0,
        )
    )


def _plot_wake_onset(wake_onset, bbox: Dict, ax: plt.Axes):
    # Wake Onset vline
    ax.vlines([wake_onset], 0, 1, transform=ax.get_xaxis_transform(), linewidth=3, linestyles='--',
              colors=colors.fau_color('nat'), zorder=3)

    # Wake Onset Text + Arrow
    ax.annotate(
        "Wake Onset",
        xy=(mdates.date2num(wake_onset), 0.90),
        xycoords=mtrans.blended_transform_factory(ax.transData, ax.transAxes),
        xytext=(mdates.date2num(wake_onset - pd.Timedelta("20M")), 0.90),
        textcoords=mtrans.blended_transform_factory(ax.transData, ax.transAxes),
        ha='right',
        va='center',
        bbox=bbox,
        size=14,
        arrowprops=dict(
            arrowstyle="->",
            lw=2,
            color=colors.fau_color('nat'),
            shrinkA=0.0, shrinkB=0.0,
        )
    )


def _plot_mrp_start(sleep_onset, mrp_start, bbox: Dict, ax: plt.Axes):
    # MRP Start vline
    ax.vlines([mrp_start], 0, 1, transform=ax.get_xaxis_transform(), linewidth=3, linestyles='--',
              colors=colors.fau_color('med'), zorder=3)
    # MRP Start Text + Arrow
    ax.annotate(
        "MRP Start",
        xy=(mdates.date2num(mrp_start), 0.80),
        xycoords=mtrans.blended_transform_factory(ax.transData, ax.transAxes),
        xytext=(mdates.date2num(sleep_onset + pd.Timedelta("20M")), 0.80),
        textcoords=mtrans.blended_transform_factory(ax.transData, ax.transAxes),
        ha='left',
        va='center',
        bbox=bbox,
        size=14,
        arrowprops=dict(
            arrowstyle="->",
            lw=2,
            color=colors.fau_color('med'),
            shrinkA=0.0, shrinkB=0.0,
        )
    )


def _plot_mrp_end(wake_onset, mrp_end, bbox: Dict, ax: plt.Axes):
    # MRP End vline
    ax.vlines([mrp_end], 0, 1, transform=ax.get_xaxis_transform(), linewidth=3, linestyles='--',
              colors=colors.fau_color('med'), zorder=3)
    # MRP End Text + Arrow
    ax.annotate(
        "MRP End",
        xy=(mdates.date2num(mrp_end), 0.80),
        xycoords=mtrans.blended_transform_factory(ax.transData, ax.transAxes),
        xytext=(mdates.date2num(wake_onset - pd.Timedelta("20M")), 0.80),
        textcoords=mtrans.blended_transform_factory(ax.transData, ax.transAxes),
        ha='right',
        va='center',
        bbox=bbox,
        size=14,
        arrowprops=dict(
            arrowstyle="->",
            lw=2,
            color=colors.fau_color('med'),
            shrinkA=0.0, shrinkB=0.0,
        )
    )


def _plot_sleep_wake_bouts(sleep_bouts: pd.DataFrame, wake_bouts: pd.DataFrame, ax: plt.Axes) -> Dict:
    handles = {}
    for (bout_name, bouts), bg_color, bg_alpha in zip({'sleep': sleep_bouts, 'wake': wake_bouts}.items(),
                                                      sleep_imu_plot_params['background.color'],
                                                      sleep_imu_plot_params['background.alpha']):
        handle = None
        for idx, bout in bouts.iterrows():
            handle = ax.axvspan(bout['start'], bout['end'], color=bg_color, alpha=bg_alpha)

        handles[bout_name] = handle

    handles = {k: v for k, v in handles.items() if v is not None}
    return handles
