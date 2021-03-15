from pathlib import Path
from typing import Sequence, Optional, Tuple, Dict, Union

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from biopsykit.carwatch_logs import LogData
from biopsykit._types import path_t
from biopsykit.io.carwatch_logs import load_logs_all_subjects


class LogStatistics:

    def __init__(self, path: path_t):
        self.path: Path = Path(path)
        self.log_dict: Dict[str, pd.DataFrame] = load_logs_all_subjects(path, return_df=False)
        self.log_data: Sequence[LogData] = [LogData(df) for df in self.log_dict.values()]

    def conditions(self) -> pd.DataFrame:
        series = pd.Series([log.condition for log in self.log_data], name="count")
        df = series.value_counts()
        df = df.reset_index().rename({'index': 'condition'}, axis=1)
        return df

    def android_versions(self, skip_na: Optional[bool] = True) -> pd.DataFrame:
        version_list = [log.android_version for log in self.log_data]
        hist = np.bincount(version_list, minlength=30)
        df = pd.DataFrame(data=hist, columns=['count'], index=range(0, len(hist)))
        # remove 1 - 20 as minimum supported android version is SDK level 21
        df.drop(list(range(1, 21)), axis=0, inplace=True)

        if skip_na:
            df.drop(0, axis=0, inplace=True)
        else:
            df.rename({0: "n/a"}, inplace=True)

        df = df.reset_index().rename({'index': 'android_version'}, axis=1)
        return df

    def app_versions(self) -> pd.DataFrame:
        series = pd.Series([log.app_version for log in self.log_data], name="count")

        df = series.value_counts().reset_index().rename({'index': 'app_version'}, axis=1)
        # df.sort_values(by=['count', 'app_version'], ascending=[False, True], inplace=True)
        return df

    def manufacturer(self, skip_na: Optional[bool] = True) -> pd.DataFrame:
        series = pd.Series([log.manufacturer for log in self.log_data], name="count")
        if skip_na:
            series = series[~series.str.contains("n/a")]

        df = series.value_counts().reset_index().rename({'index': 'manufacturer'}, axis=1)
        df.sort_values(by=['count', 'manufacturer'], ascending=[False, True], inplace=True)
        return df

    def models(self, skip_na: Optional[bool] = True) -> pd.DataFrame:
        series = pd.Series([log.model for log in self.log_data], name="count")
        if skip_na:
            series = series[~series.str.contains("n/a")]
        df: pd.DataFrame = series.value_counts()

        df = df.reset_index().rename({'index': 'model'}, axis=1)
        df.sort_values(by=['count', 'model'], ascending=[False, True], inplace=True)
        return df

    def finished_days(self) -> pd.DataFrame:
        series = pd.Series([log.num_finished_days for log in self.log_data], name="count")
        df = series.value_counts(sort=False)

        df = df.reset_index().rename({'index': 'finished_days'}, axis=1)
        return df

    def days(self) -> pd.DataFrame:
        series = pd.Series(np.concatenate([log.log_dates for log in self.log_data]), name="count")
        df = series.value_counts(sort=False)
        df.sort_index(inplace=True)

        df = df.reset_index().rename({'index': 'logging_days'}, axis=1)
        return df

    def get_plot(self, plot_id: str, ax: Optional[plt.Axes] = None) -> Union[Tuple[plt.Figure, plt.Axes], None]:
        import seaborn as sns

        fig: Union[plt.Figure, None] = None
        if ax is None:
            fig, ax = plt.subplots()

        if plot_id in ['condition', 'conditions']:
            df = self.conditions()
        elif plot_id in ['manufacturer', 'manufacturers']:
            df = self.manufacturer()
        elif plot_id in ['android', 'android_version', 'android_versions']:
            df = self.android_versions()
        elif plot_id in ['app', 'app_version', 'app_versions']:
            df = self.app_versions()
        elif plot_id in ['model', 'models']:
            df = self.models()
        elif plot_id in ['finished', 'finished_days']:
            df = self.finished_days()
        elif plot_id in ['days', 'logging_days']:
            df = self.days()
        else:
            raise ValueError("Invalid plot_id '{}'!".format(plot_id))

        palette = sns.cubehelix_palette(len(df), start=.5, rot=-.75)

        cols = df.columns
        ax = sns.barplot(x=cols[0], y=cols[1], data=df, ax=ax, palette=palette)

        if plot_id in ['model', 'days']:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        if fig:
            return fig, ax
