"""Module with utility functions for machine learning and classification applications."""
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

__all__ = ["factorize_subject_id", "prepare_df_sklearn", "split_train_test", "strip_df", "strip_labels"]


class _PipelineWrapper:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def __str__(self):
        return str(self.pipeline)

    def __repr__(self):
        return repr(self.pipeline)


# TODO check if works?
# def __getitem__(self, item):
#     return self.pipeline[item]
#
# def __len__(self):
#     return len(self.pipeline)


def strip_df(data: pd.DataFrame) -> np.ndarray:
    """Strip dataframe from all index levels to only contain values.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        input dataframe

    Returns
    -------
    :class:`~numpy.ndarray`
        array of stripped dataframe without index

    """
    return np.array(data.reset_index(drop=True).values)


def strip_labels(data: Union[pd.DataFrame, pd.Series], label_col: Optional[str] = None) -> np.ndarray:
    """Strip labels from dataframe index.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame` or :class:`~pandas.Series`
        input data
    label_col : str, optional
        name of index level containing class labels or ``None`` to use default column name ("label").
        Default: ``None``

    Returns
    -------
    :class:`~numpy.ndarray`
        array with labels

    """
    # TODO change to dataframe column
    if label_col is None:
        label_col = "label"
    if isinstance(data, pd.DataFrame):
        data = data.index.get_level_values(label_col)
    return np.array(data)


def factorize_subject_id(
    data: Union[pd.Series, pd.DataFrame], subject_col: Optional[str] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Factorize subject IDs, i.e., encode them as an enumerated type or categorical variable.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame` or :class:`~pandas.Series`
        input data
    subject_col : str, optional
        name of index level containing subject IDs or ``None`` to use default column name ("subject").
        Default: ``None``

    Returns
    -------
    groups : :class:`~numpy.ndarray`
        A numpy array with factorized subject IDs. They also serve as indexer for ``keys``.
    keys : :class:`~numpy.ndarray`
        The unique subject ID values.

    """
    if subject_col is None:
        subject_col = "subject"

    if isinstance(data, pd.DataFrame):
        data = data.index.get_level_values(subject_col)
    groups, keys = pd.factorize(data)
    return groups, keys


def prepare_df_sklearn(
    data: pd.DataFrame,
    label_col: Optional[str] = None,
    subject_col: Optional[str] = None,
    print_summary: Optional[bool] = False,
) -> tuple[np.ndarray, ...]:
    """Prepare a dataframe for usage in sklearn functions and return the single components of the dataframe.

    This function performs the following steps:

    * Strip dataframe from all index levels and return an array that only contains values
      (using :func:`~biopsykit.classification.utils.strip_df`)
    * Extract labels from dataframe (using :func:`~biopsykit.classification.utils.strip_labels`)
    * Factorize subject IDs so that each subject ID has an unique number
      (using :func:`~biopsykit.classification.utils.factorize_subject_id`)


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        Input data as pandas dataframe
    label_col : str, optional
        name of index level containing class labels or ``None`` to use default column name ("label").
        Default: ``None``
    subject_col : str, optional
        name of index level containing subject IDs or ``None`` to use default column name ("subject").
        Default: ``None``
    print_summary : bool, optional
        ``True`` to print a summary of the shape of the data and label arrays, the number of groups and the class
        prevalence of all classes, ``False`` otherwise.
        Default: ``False``

    Returns
    -------
    X : array-like of shape (`n_samples`, `n_features`)
        Training vector, where `n_samples` is the number of samples and `n_features` is the number of features.
    y_data : array-like of shape (`n_samples`,)
        Target relative to ``X``, i.e. class labels.
    groups : array-like of shape (`n_samples`,)
        Factorized subject IDs
    group_keys : array-like of shape (`n_samples`,)
        Subject IDs

    """
    x_data = strip_df(data)
    y_data = strip_labels(data, label_col)
    groups, group_keys = factorize_subject_id(data, subject_col)

    if print_summary:
        print(
            f"Shape of X: {x_data.shape}; shape of y: {y_data.shape}; "
            f"number of groups: {len(group_keys)}, class prevalence: {np.unique(y_data, return_counts=True)[1]}"
        )

    return x_data, y_data, groups, group_keys


def split_train_test(
    X: np.ndarray,  # noqa: N803
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, ...]:
    """Split data into train and test set.

    Parameters
    ----------
    X : array-like of shape (`n_samples`, `n_features`)
        Data to be split, where `n_samples` is the number of samples and `n_features` is the number of features.
    y : array-like of shape (`n_samples`,)
        Target relative to ``x_data``, i.e. class labels.
    train : :class:`~numpy.ndarray`
        The training set indices for that split
    test : :class:`~numpy.ndarray`
        The test set indices for that split
    groups : array-like of shape (`n_samples`,), optional
        Group labels for the samples used while splitting the dataset into train/test set or ``None`` if group labels
        should not be considered for splitting.
        Default: ``None``


    Returns
    -------
    X_train: :class:`~numpy.ndarray`
        Training data
    X_test: :class:`~numpy.ndarray`
        Test data
    y_train: :class:`~numpy.ndarray`
        Targets of training data
    y_test: :class:`~numpy.ndarray`
        Targets of test data
    group_train: :class:`~numpy.ndarray`
        Group labels of training data (only available if ``groups`` is not ``None``)
    group_test: :class:`~numpy.ndarray`
        Group labels of test data (only available if ``groups`` is not ``None``)

    """
    X_train, X_test = X[train], X[test]  # noqa: N806
    y_train, y_test = y[train], y[test]

    if groups is None:
        return X_train, X_test, y_train, y_test

    groups_train = groups[train]
    groups_test = groups[test]
    return X_train, X_test, y_train, y_test, groups_train, groups_test


def merge_nested_dicts(dict1: dict, dict2: dict) -> dict:
    """Merge two nested dictionaries.

    Parameters
    ----------
    dict1 : dict
        First dictionary to merge
    dict2 : dict
        Second dictionary to merge

    Returns
    -------
    dict
        Merged dictionary

    """
    dict1 = deepcopy(dict1)
    return _merge_nested_dicts(dict1, dict2)


def _merge_nested_dicts(dict1: dict, dict2: dict) -> dict:
    for key, value in dict2.items():
        if isinstance(value, dict) and key in dict1:
            _merge_nested_dicts(dict1[key], value)
            # check if value is list
        elif isinstance(value, list) and key in dict1:
            dict1[key] = value if key not in dict1 else dict1[key] + value
            list_of_dicts = deepcopy(dict1[key])
            merged_dict = {}

            for d in list_of_dicts:
                for k, v in d.items():
                    # Use set to avoid duplicates, then convert it back to a list
                    merged_dict[k] = list(set(merged_dict.get(k, []) + list(v)))

            # Convert the merged result back to dictionaries
            result = [dict(merged_dict)]
            dict1[key] = result
        elif key not in dict1:
            dict1[key] = value
    return dict1
