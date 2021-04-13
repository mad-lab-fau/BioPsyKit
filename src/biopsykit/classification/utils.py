from typing import Union, Tuple, Optional

import pandas as pd
import numpy as np


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
    X = strip_df(data)
    y = strip_labels(data)
    groups, group_keys = factorize_subject_id(data)

    if print_summary:
        print(
            "Shape of X: {}; shape of y: {}; number of groups: {}, class prevalences: {}".format(
                X.shape, y.shape, len(group_keys), np.unique(y, return_counts=True)[1]
            )
        )

    return X, y, groups, group_keys


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, ...]:
    Xtrain, Xtest = X[train], X[test]
    ytrain, ytest = y[train], y[test]

    if groups is not None:
        groups_train = groups[train]
        groups_test = groups[test]
        return Xtrain, Xtest, ytrain, ytest, groups_train, groups_test

    return Xtrain, Xtest, ytrain, ytest
