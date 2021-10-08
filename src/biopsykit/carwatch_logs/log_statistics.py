"""Module for generating log data statistics."""
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from biopsykit.carwatch_logs import LogData
from biopsykit.io.carwatch_logs import load_logs_all_subjects
from biopsykit.utils._types import path_t


class LogStatistics:
    """Class to compute statistics from CARWatch log data collected during one study."""

    def __init__(self, base_folder: path_t):
        """Initialize a new ``LogStatistics`` instance.

        Parameters
        ----------
        base_folder : :class:`~pathlib.Path` or str
            base folder path to log data from all subjects of one study

        """
        self.path: Path = Path(base_folder)
        self.log_dict: Dict[str, pd.DataFrame] = load_logs_all_subjects(self.path, return_df=False)
        self.log_data: Sequence[LogData] = [LogData(df) for df in self.log_dict.values()]

    def conditions(self) -> pd.DataFrame:
        """Return statistics of study conditions available in the log data.

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with study conditions and their frequency

        """
        series = pd.Series([log.condition for log in self.log_data], name="count")
        df = series.value_counts()
        df = df.reset_index().rename({"index": "condition"}, axis=1)
        return df

    def android_versions(self, skip_na: Optional[bool] = True) -> pd.DataFrame:
        """Return statistics of Android versions of the smartphones used in the study.

        Parameters
        ----------
        skip_na : bool, optional
            ``True`` to exclude Android versions that are not present in the study from the result dataframe,
            ``False`` otherwise. Default: ``True``

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with all available Android versions and their frequency

        """
        version_list = [log.android_version for log in self.log_data]
        hist = np.bincount(version_list, minlength=30)
        df = pd.DataFrame(data=hist, columns=["count"], index=range(0, len(hist)))
        # remove 1 - 20 as minimum supported android version is SDK level 21
        df.drop(list(range(1, 21)), axis=0, inplace=True)

        if skip_na:
            df.drop(0, axis=0, inplace=True)
        else:
            df.rename({0: "n/a"}, inplace=True)

        df = df.reset_index().rename({"index": "android_version"}, axis=1)
        return df

    def app_versions(self) -> pd.DataFrame:
        """Return statistics of CARWatch App versions used in the study.

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with all available CARWatch App versions and their frequency

        """
        series = pd.Series([log.app_version for log in self.log_data], name="count")

        df = series.value_counts().reset_index().rename({"index": "app_version"}, axis=1)
        # df.sort_values(by=['count', 'app_version'], ascending=[False, True], inplace=True)
        return df

    def manufacturer(self, skip_na: Optional[bool] = True) -> pd.DataFrame:
        """Return statistics of the manufacturer names of smartphones used in the study.

        Parameters
        ----------
        skip_na : bool, optional
            ``True`` to exclude non-available manufacturer names from the result dataframe,
            ``False`` otherwise. Default: ``True``

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with all available smartphone manufacturers and their frequency

        """
        series = pd.Series([log.manufacturer for log in self.log_data], name="count")
        if skip_na:
            series = series[~series.str.contains("n/a")]

        df = series.value_counts().reset_index().rename({"index": "manufacturer"}, axis=1)
        df.sort_values(by=["count", "manufacturer"], ascending=[False, True], inplace=True)
        return df

    def models(self, skip_na: Optional[bool] = True) -> pd.DataFrame:
        """Return statistics of the smartphone models used in the study.

        Parameters
        ----------
        skip_na : bool, optional
            ``True`` to exclude non-available smartphone model names from the result dataframe,
            ``False`` otherwise. Default: ``True``

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with all available smartphone model names and their frequency

        """
        series = pd.Series([log.model for log in self.log_data], name="count")
        if skip_na:
            series = series[~series.str.contains("n/a")]
        df: pd.DataFrame = series.value_counts()

        df = df.reset_index().rename({"index": "model"}, axis=1)
        df.sort_values(by=["count", "model"], ascending=[False, True], inplace=True)
        return df

    def finished_days(self) -> pd.DataFrame:
        """Return statistics of finished study days per subject.

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with frequency of finished study days

        """
        series = pd.Series([log.num_finished_days for log in self.log_data], name="count")
        df = series.value_counts(sort=False)

        df = df.reset_index().rename({"index": "finished_days"}, axis=1)
        return df

    def days(self) -> pd.DataFrame:
        """Return statistics of recording days in the study.

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with frequency of recording days

        """
        series = pd.Series(np.concatenate([log.log_dates for log in self.log_data]), name="count")
        df = series.value_counts(sort=False)
        df.sort_index(inplace=True)

        df = df.reset_index().rename({"index": "logging_days"}, axis=1)
        return df

    def get_plot(self, plot_id: str, **kwargs) -> Tuple[plt.Figure, plt.Axes]:  # pylint:disable=too-many-branches
        """Return barplot to visualize log data statistics for one data type.

        Parameters
        ----------
        plot_id : str
            type of data to plot

        Returns
        -------
        fig : :class:`matplotlib.figure.Figure`
            figure object
        ax : :class:`matplotlib.axes.Axes`
            axes object

        """
        ax: plt.Axes = kwargs.pop("ax", None)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if plot_id in ["condition", "conditions"]:
            df = self.conditions()
        elif plot_id in ["manufacturer", "manufacturers"]:
            df = self.manufacturer()
        elif plot_id in ["android", "android_version", "android_versions"]:
            df = self.android_versions()
        elif plot_id in ["app", "app_version", "app_versions"]:
            df = self.app_versions()
        elif plot_id in ["model", "models"]:
            df = self.models()
        elif plot_id in ["finished", "finished_days"]:
            df = self.finished_days()
        elif plot_id in ["days", "logging_days"]:
            df = self.days()
        else:
            raise ValueError("Invalid plot_id '{}'!".format(plot_id))

        palette = sns.cubehelix_palette(len(df), start=0.5, rot=-0.75)

        cols = df.columns
        ax = sns.barplot(x=cols[0], y=cols[1], data=df, ax=ax, palette=palette)

        if plot_id in ["model", "days"]:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        return fig, ax
