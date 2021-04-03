from typing import Union, Optional, Dict, Tuple, Sequence

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import biopsykit.colors as colors

sleep_imu_plot_params = {
    "background.color": ["#e0e0e0", "#9e9e9e"],
    "background.alpha": [0.3, 0.3],
}

_bbox_default = dict(
    fc=(1, 1, 1, plt.rcParams["legend.framealpha"]),
    ec=plt.rcParams["legend.edgecolor"],
    boxstyle="round",
)


def sleep_imu_plot(
    data: pd.DataFrame,
    datastreams: Optional[Union[str, Sequence[str]]] = None,
    sleep_endpoints: Optional[Union[Dict, pd.DataFrame]] = None,
    downsample_factor: Optional[float] = 1,
    **kwargs,
) -> Union[Tuple[plt.Figure, plt.Axes], None]:
    import matplotlib.ticker as mticks

    fig: Union[plt.Figure, None] = None
    axs: Union[plt.Axes, Sequence[plt.Axes], None] = kwargs.get("ax", kwargs.get("axs", None))
    sns.set_palette(colors.fau_palette_blue("line_3"))

    downsample_factor = int(downsample_factor)
    if downsample_factor < 1:
        raise ValueError("Invalid downsample factor!")
    if datastreams is None:
        datastreams = ["acc"]
    if isinstance(datastreams, str):
        datastreams = [datastreams]

    if axs is None:
        figsize = kwargs.get("figsize", plt.rcParams["figure.figsize"])
        nrows = len(datastreams)
        fig, axs = plt.subplots(figsize=figsize, nrows=nrows)

    if isinstance(axs, plt.Axes):
        axs = [axs]

    if len(datastreams) != len(axs):
        raise ValueError("Number of datastreams to be plotted must match number of supplied subplots!")

    if isinstance(data.index, pd.DatetimeIndex):
        plt.rcParams["timezone"] = data.index.tz.zone

    ylabel = kwargs.get(
        "ylabel",
        {
            "acc": "Acceleration [$m/s^2$]",
            "gyr": "Angular Velocity [$°/s$]",
        },
    )
    xlabel = kwargs.get("xlabel", "Time")

    for ax, ds in zip(axs, datastreams):
        data_plot = data.filter(like=ds)[::downsample_factor]
        data_plot.plot(ax=ax)
        if sleep_endpoints is not None:
            kwargs["ax"] = ax
            _plot_sleep_endpoints(sleep_endpoints=sleep_endpoints, **kwargs)

        if isinstance(data_plot.index, pd.DatetimeIndex):
            # TODO add axis style for non-Datetime axes
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.xaxis.set_minor_locator(mticks.AutoMinorLocator(6))

        ax.set_ylabel(ylabel[ds])
        ax.set_xlabel(xlabel)
        ax.legend(loc="lower left", framealpha=1.0)

    if fig:
        fig.tight_layout()
        fig.autofmt_xdate(rotation=0, ha="center")
        return fig, axs


def _plot_sleep_endpoints(sleep_endpoints: Dict, **kwargs):
    mrp_start = pd.to_datetime(sleep_endpoints["major_rest_period_start"])
    mrp_end = pd.to_datetime(sleep_endpoints["major_rest_period_end"])
    sleep_onset = pd.to_datetime(sleep_endpoints["sleep_onset"])
    wake_onset = pd.to_datetime(sleep_endpoints["wake_onset"])

    ax = kwargs.get("ax")

    plot_sleep_onset = kwargs.get("plot_sleep_onset", True)
    plot_wake_onset = kwargs.get("plot_wake_onset", True)
    plot_mrp_start = kwargs.get("plot_mrp_start", True)
    plot_mrp_end = kwargs.get("plot_mrp_end", True)
    plot_sleep_wake = kwargs.get("plot_sleep_wake", True)

    if isinstance(sleep_endpoints, dict):
        sleep_bouts = sleep_endpoints["sleep_bouts"]
        wake_bouts = sleep_endpoints["wake_bouts"]
        date = sleep_endpoints["date"]
    else:
        sleep_bouts = pd.DataFrame(sleep_endpoints["sleep_bouts"][0])
        wake_bouts = pd.DataFrame(sleep_endpoints["wake_bouts"][0])
        date = sleep_endpoints.index[0][1]

    date = pd.to_datetime(date)

    # 00:00 (12 am) vline (if present)
    if date == mrp_start.normalize():
        ax.vlines(
            [date + pd.Timedelta("1d")],
            0,
            1,
            transform=ax.get_xaxis_transform(),
            linewidths=3,
            linestyles="dotted",
            colors=colors.fau_color("tech"),
            zorder=0,
        )

    if plot_sleep_onset:
        _plot_sleep_onset(sleep_onset, **kwargs)
    if plot_wake_onset:
        _plot_wake_onset(wake_onset, **kwargs)
    if plot_mrp_start:
        _plot_mrp_start(sleep_onset, mrp_start, **kwargs)
    if plot_mrp_end:
        _plot_mrp_end(wake_onset, mrp_end, **kwargs)
    if plot_sleep_wake:
        handles = _plot_sleep_wake_bouts(sleep_bouts, wake_bouts, ax)
        legend = ax.legend(
            handles=list(handles.values()),
            labels=list(handles.keys()),
            loc="lower right",
            framealpha=1.0,
            fontsize=14,
        )
        ax.add_artist(legend)

    # wear_time['end'] = wear_time.index.shift(1, freq=pd.Timedelta("15M"))
    # wear_time = wear_time[wear_time['wear'] == 0.0]
    # wear_time = wear_time.reset_index()
    #
    # handle = None
    # for idx, row in wear_time.iterrows():
    #     handle = ax.axvspan(row['index'], row['end'], color=colors.fau_color('wiso'), alpha=0.5, lw=0)
    # if handle is not None:
    #     handles['non-wear'] = handle

    ax.set_title("Sleep IMU Data: {} – {}".format(date.date(), (date + pd.Timedelta("1d")).date()))


def _plot_sleep_onset(sleep_onset, **kwargs):
    ax = kwargs.get("ax")
    bbox = kwargs.get("bbox", _bbox_default)

    # Sleep Onset vline
    ax.vlines(
        [sleep_onset],
        0,
        1,
        transform=ax.get_xaxis_transform(),
        linewidth=3,
        linestyles="--",
        colors=colors.fau_color("nat"),
        zorder=3,
    )

    # Sleep Onset Text + Arrow
    ax.annotate(
        "Sleep Onset",
        xy=(mdates.date2num(sleep_onset), 0.90),
        xycoords=ax.get_xaxis_transform(),
        xytext=(mdates.date2num(sleep_onset + pd.Timedelta("20min")), 0.90),
        textcoords=ax.get_xaxis_transform(),
        ha="left",
        va="center",
        bbox=bbox,
        size=14,
        arrowprops=dict(
            arrowstyle="->",
            lw=2,
            color=colors.fau_color("nat"),
            shrinkA=0.0,
            shrinkB=0.0,
        ),
    )


def _plot_wake_onset(wake_onset, **kwargs):
    ax = kwargs.get("ax")
    bbox = kwargs.get("bbox", _bbox_default)
    # Wake Onset vline
    ax.vlines(
        [wake_onset],
        0,
        1,
        transform=ax.get_xaxis_transform(),
        linewidth=3,
        linestyles="--",
        colors=colors.fau_color("nat"),
        zorder=3,
    )

    # Wake Onset Text + Arrow
    ax.annotate(
        "Wake Onset",
        xy=(mdates.date2num(wake_onset), 0.90),
        xycoords=ax.get_xaxis_transform(),
        xytext=(mdates.date2num(wake_onset - pd.Timedelta("20min")), 0.90),
        textcoords=ax.get_xaxis_transform(),
        ha="right",
        va="center",
        bbox=bbox,
        size=14,
        arrowprops=dict(
            arrowstyle="->",
            lw=2,
            color=colors.fau_color("nat"),
            shrinkA=0.0,
            shrinkB=0.0,
        ),
    )


def _plot_mrp_start(sleep_onset, mrp_start, **kwargs):
    ax = kwargs.get("ax")
    bbox = kwargs.get("bbox", _bbox_default)

    # MRP Start vline
    ax.vlines(
        [mrp_start],
        0,
        1,
        transform=ax.get_xaxis_transform(),
        linewidth=3,
        linestyles="--",
        colors=colors.fau_color("med"),
        zorder=3,
    )
    # MRP Start Text + Arrow
    ax.annotate(
        "MRP Start",
        xy=(mdates.date2num(mrp_start), 0.80),
        xycoords=ax.get_xaxis_transform(),
        xytext=(mdates.date2num(sleep_onset + pd.Timedelta("20min")), 0.80),
        textcoords=ax.get_xaxis_transform(),
        ha="left",
        va="center",
        bbox=bbox,
        size=14,
        arrowprops=dict(
            arrowstyle="->",
            lw=2,
            color=colors.fau_color("med"),
            shrinkA=0.0,
            shrinkB=0.0,
        ),
    )


def _plot_mrp_end(wake_onset, mrp_end, **kwargs):
    ax = kwargs.get("ax")
    bbox = kwargs.get("bbox", _bbox_default)

    # MRP End vline
    ax.vlines(
        [mrp_end],
        0,
        1,
        transform=ax.get_xaxis_transform(),
        linewidth=3,
        linestyles="--",
        colors=colors.fau_color("med"),
        zorder=3,
    )
    # MRP End Text + Arrow
    ax.annotate(
        "MRP End",
        xy=(mdates.date2num(mrp_end), 0.80),
        xycoords=ax.get_xaxis_transform(),
        xytext=(mdates.date2num(wake_onset - pd.Timedelta("20min")), 0.80),
        textcoords=ax.get_xaxis_transform(),
        ha="right",
        va="center",
        bbox=bbox,
        size=14,
        arrowprops=dict(
            arrowstyle="->",
            lw=2,
            color=colors.fau_color("med"),
            shrinkA=0.0,
            shrinkB=0.0,
        ),
    )


def _plot_sleep_wake_bouts(sleep_bouts: pd.DataFrame, wake_bouts: pd.DataFrame, ax: plt.Axes) -> Dict:
    handles = {}
    for (bout_name, bouts), bg_color, bg_alpha in zip(
        {"sleep": sleep_bouts, "wake": wake_bouts}.items(),
        sleep_imu_plot_params["background.color"],
        sleep_imu_plot_params["background.alpha"],
    ):
        handle = None
        for idx, bout in bouts.iterrows():
            handle = ax.axvspan(bout["start"], bout["end"], color=bg_color, alpha=bg_alpha)

        handles[bout_name] = handle

    handles = {k: v for k, v in handles.items() if v is not None}
    return handles
