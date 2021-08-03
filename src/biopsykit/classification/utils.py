"""Module with utility functions for machine learning and classification applications."""
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
    if label_col is None:
        label_col = "label"
    if isinstance(data, pd.DataFrame):
        data = data.index.get_level_values(label_col)
    return np.array(data)


def factorize_subject_id(
    data: Union[pd.Series, pd.DataFrame], subject_col: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
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
) -> Tuple[np.ndarray, ...]:
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
            "Shape of X: {}; shape of y: {}; number of groups: {}, class prevalence: {}".format(
                x_data.shape, y_data.shape, len(group_keys), np.unique(y_data, return_counts=True)[1]
            )
        )

    return x_data, y_data, groups, group_keys


def split_train_test(  # pylint:disable=invalid-name
    X: np.ndarray,  # noqa
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, ...]:
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
    X_train, X_test = X[train], X[test]  # noqa
    y_train, y_test = y[train], y[test]

    if groups is None:
        return X_train, X_test, y_train, y_test

    groups_train = groups[train]
    groups_test = groups[test]
    return X_train, X_test, y_train, y_test, groups_train, groups_test
