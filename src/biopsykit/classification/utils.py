from typing import Union, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

__all__ = ["factorize_subject_id", "prepare_df_sklearn", "split_train_test", "strip_df", "strip_labels"]


class _PipelineWrapper:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def __str__(self):
        return str(self.pipeline)

    def __repr__(self):
        return repr(self.pipeline)


def strip_df(data: pd.DataFrame) -> np.ndarray:
    return data.reset_index(drop=True).values


def strip_labels(data: Union[pd.DataFrame, pd.Series]) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        data = data.index.get_level_values("label")
    return np.array(data)


def factorize_subject_id(data: Union[pd.Series, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(data, pd.DataFrame):
        data = data.index.get_level_values("subject")
    groups, keys = pd.factorize(data)
    return groups, keys


def prepare_df_sklearn(data: pd.DataFrame, print_summary: Optional[bool] = False) -> Tuple:
    x = strip_df(data)
    y = strip_labels(data)
    groups, group_keys = factorize_subject_id(data)

    if print_summary:
        print(
            "Shape of X: {}; shape of y: {}; number of groups: {}, class prevalence: {}".format(
                x.shape, y.shape, len(group_keys), np.unique(y, return_counts=True)[1]
            )
        )

    return x, y, groups, group_keys


def split_train_test(
    x_data: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, ...]:
    x_train, x_test = x_data[train], x_data[test]
    y_train, y_test = y[train], y[test]

    if groups is not None:
        groups_train = groups[train]
        groups_test = groups[test]
        return x_train, x_test, y_train, y_test, groups_train, groups_test

    return x_train, x_test, y_train, y_test
