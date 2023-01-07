"""Functions to analyze classification results."""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from fau_colors import cmaps
from matplotlib import pyplot as plt
from matplotlib.cm import register_cmap
from matplotlib.colors import ListedColormap
from sklearn.metrics import ConfusionMatrixDisplay

from biopsykit.classification.model_selection import SklearnPipelinePermuter
from biopsykit.classification.utils import prepare_df_sklearn


def predictions_as_df(
    pipeline_permuter: SklearnPipelinePermuter,
    data: pd.DataFrame,
    pipeline: Tuple[str],
    label_mapping: Optional[Dict[str, str]] = None,
    index_col: Optional[str] = None,
) -> pd.DataFrame:
    """Get predictions from a specified pipeline and merge them with the index of the input dataframe.

    Parameters
    ----------
    pipeline_permuter : :class:`~biopsykit.classification.model_selection.SklearnPipelinePermuter`
        :class:`~biopsykit.classification.model_selection.SklearnPipelinePermuter` instance
    data : :class:`~pandas.DataFrame`
        input data
    pipeline : tuple
        pipeline to get predictions from
    label_mapping : dict, optional
        mapping of labels to rename labels in the output dataframe or ``None`` to keep original labels.
        Default: ``None``
    index_col : str, optional
        name of the index column to merge the predictions with. If ``data`` has a multi-index,
        the first level is used unless ``index_col`` is specified.
        Default: ``None``

    Returns
    -------
    :class:`~pandas.DataFrame`
        predictions as dataframe

    """
    metric_summary = pipeline_permuter.metric_summary()
    label_cols = ["true_labels", "predicted_labels"]
    predictions = metric_summary[label_cols].explode(label_cols).loc[pipeline]

    if isinstance(data.index, pd.MultiIndex):
        if index_col is None:
            index_col = data.index.names[0]
        index_vals = data.index.get_level_values(index_col)
    else:
        index_vals = data.index

    predictions.index = index_vals
    if label_mapping:
        predictions = predictions.replace(label_mapping)
    return predictions


def predict_proba_from_estimator(
    pipeline_permuter: SklearnPipelinePermuter,
    data: pd.DataFrame,
    pipeline: Tuple[str],
    label_col: Optional[str] = "label",
    column_names: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Get predictions as probabilities from a specified pipeline and merge them with the index of the input dataframe.

    Parameters
    ----------
    pipeline_permuter : :class:`~biopsykit.classification.model_selection.SklearnPipelinePermuter`
        :class:`~biopsykit.classification.model_selection.SklearnPipelinePermuter` instance
    data : :class:`~pandas.DataFrame`
        input data
    pipeline : tuple
        pipeline to get predictions from
    label_col : str, optional
        name of the label column in the input dataframe. Default: ``"label"``
    column_names : dict, optional
        mapping of column names to rename columns in the output dataframe or ``None`` to keep original column names.
        Default: ``None``

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with predictions as probabilities

    """
    metric_summary = pipeline_permuter.metric_summary()
    best_pipeline = pipeline_permuter.best_estimator_summary()
    best_pipeline = best_pipeline.loc[pipeline].iloc[0].pipeline

    test_indices = metric_summary.loc[pipeline]["test_indices_folds"]
    test_indices_flat = list(metric_summary.loc[pipeline]["test_indices"])

    x, y, _, _ = prepare_df_sklearn(data, label_col=label_col, print_summary=False)

    label_order = best_pipeline[0].classes_

    predict_proba_results = []
    predict_proba_labels = []

    for i, test_idx in enumerate(test_indices):
        test_idx = list(test_idx)
        pipeline_fold = best_pipeline[i]
        predict_proba_results.append(pipeline_fold.predict_proba(x[test_idx]))
        predict_proba_labels.append(y[test_idx])

    results_proba = pd.DataFrame(
        np.concatenate(predict_proba_results),
        columns=label_order,
        index=data.index[test_indices_flat],
    )
    if column_names is not None:
        results_proba = results_proba.rename(columns=column_names)
    results_proba = results_proba.sort_index()
    results_proba = results_proba.round(4)

    return results_proba


def _conf_matrix_from_proba_df(
    data: pd.DataFrame, label_col: Optional[str] = "label", label_order: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Get confusion matrix from a dataframe with predictions as probabilities.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe with predictions as probabilities
    label_col : str, optional
        name of the label column in the input dataframe. Default: ``"label"``
    label_order : list, optional
        order of labels in the confusion matrix or ``None`` to use the order of the labels in the input dataframe.
        Default: ``None``

    Returns
    -------
    :class:`~pandas.DataFrame`
        confusion matrix as dataframe

    """
    if label_order is None:
        label_order = list(data.columns)
    conf_matrix_proba = data.groupby(label_col).mean()
    conf_matrix_proba = conf_matrix_proba.reindex(label_order, axis=0).reindex(label_order, axis=1)
    return conf_matrix_proba


def _register_fau_r():
    fau_r = sns.color_palette(cmaps.fau)
    fau_r.reverse()
    fau_r = ListedColormap(fau_r, "fau_r")
    if "fau_r" not in plt.colormaps():
        register_cmap("fau_r", fau_r)


def plot_conf_matrix(predictions: pd.DataFrame, labels: Sequence[str], **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Plot confusion matrix from predictions.

    Parameters
    ----------
    predictions : :class:`~pandas.DataFrame`
        dataframe with predictions
    labels : list
        list of labels
    **kwargs
        additional keyword arguments to pass to :func:`plt.subplots`

    """
    # check if ax is given
    ax = kwargs.get("ax", None)
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = ax.get_figure()

    # check if fau_r is registered as colormap
    if "fau_r" not in plt.colormaps():
        _register_fau_r()

    ConfusionMatrixDisplay.from_predictions(
        predictions["true_labels"],
        predictions["predicted_labels"],
        cmap="fau_r",
        labels=labels,
        colorbar=False,
        ax=ax,
    )
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_color(None)

    return fig, ax


def plot_conf_matrix_proba(predictions: pd.DataFrame, labels: Sequence[str], **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Plot confusion matrix from prediction probabilities.

    Parameters
    ----------
    predictions : :class:`~pandas.DataFrame`
        dataframe with predictions as probabilities
    labels : list
        list of labels
    **kwargs
        additional keyword arguments to pass to :func:`plt.subplots`

    """
    # check if ax is given
    ax = kwargs.get("ax", None)
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = ax.get_figure()

    # check if fau_r is registered as colormap
    if "fau_r" not in plt.colormaps():
        _register_fau_r()

    conf_matrix_proba = _conf_matrix_from_proba_df(predictions, label_order=labels)

    sns.heatmap(conf_matrix_proba, cmap="fau_r", annot=True, cbar=False, square=True, ax=ax)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    return fig, ax
