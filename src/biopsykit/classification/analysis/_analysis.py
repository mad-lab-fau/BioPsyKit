"""Functions to analyze classification results."""
from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import seaborn as sns
from fau_colors import cmaps
from matplotlib import pyplot as plt
from matplotlib.cm import ColormapRegistry
from matplotlib.colors import ListedColormap
from sklearn.metrics import ConfusionMatrixDisplay

from biopsykit.classification.model_selection import SklearnPipelinePermuter
from biopsykit.classification.utils import prepare_df_sklearn

pipeline_step_map = {
    "pipeline_scaler": "Scaler",
    "pipeline_reduce_dim": r"\makecell[lc]{Feature\\ Selection}",
    "pipeline_clf": "Classifier",
}

metric_map = {
    "accuracy": r"\makecell{Accuracy [\%]}",
    "f1": r"\makecell{F1-score [\%]}",
    "precision": r"\makecell{Precision [\%]}",
    "recall": r"\makecell{Recall [\%]}",
    "auc": r"\makecell{AUC [\%]}",
    "sensitivity": r"\makecell{Sensitivity [\%]}",
    "specificity": r"\makecell{Specificity [\%]}",
}

clf_map = {
    "MinMaxScaler": "Min-Max",
    "StandardScaler": "Standard",
    "SelectKBest": "SkB",
    "RFE": "RFE",
    "GaussianNB": "NB",
    "KNeighborsClassifier": "kNN",
    "DecisionTreeClassifier": "DT",
    "SVC": "SVM",
    "RandomForestClassifier": "RF",
    "MLPClassifier": "MLP",
    "AdaBoostClassifier": "Ada",
}


def predictions_as_df(
    pipeline_permuter: SklearnPipelinePermuter,
    data: pd.DataFrame,
    pipeline: tuple[str],
    label_mapping: Optional[dict[str, str]] = None,
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
    pipeline: tuple[str],
    label_col: Optional[str] = "label",
    column_names: Optional[dict[str, str]] = None,
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
        test_idx_list = list(test_idx)
        pipeline_fold = best_pipeline[i]
        predict_proba_results.append(pipeline_fold.predict_proba(x[test_idx_list]))
        predict_proba_labels.append(y[test_idx_list])

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
        colormap_registry = ColormapRegistry()
        colormap_registry.register(cmap=fau_r, name="fau_r")


def plot_conf_matrix(
    predictions: pd.DataFrame,
    labels: Sequence[str],
    label_name: Optional[str] = "label",
    conf_matrix_kwargs: Optional[dict[str, Any]] = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot confusion matrix from predictions.

    Parameters
    ----------
    predictions : :class:`~pandas.DataFrame`
        dataframe with predictions
    labels : list, dict, optional
        list of labels to use in the confusion matrix or dictionary with label names in the data frame as key and the
        corresponding label names to use in the confusion matrix as value.
        Default: ``None`` to use the labels in the data frame in the order they appear
    label_name : str, optional
        name of the 'label' in the axis titles. Default: "label" to yield "True label" and "Predicted label"
    conf_matrix_kwargs : dict, optional
        additional keyword arguments to pass to :func:`~sklearn.metrics.ConfusionMatrixDisplay.from_predictions`
    **kwargs
        additional keyword arguments to pass to :func:`plt.subplots`

    """
    if conf_matrix_kwargs is None:
        conf_matrix_kwargs = {}
    # check if ax is given
    ax = kwargs.get("ax")
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = ax.get_figure()

    predictions = predictions.copy()
    if isinstance(labels, dict):
        # only replace in true_labels and predicted_labels columns
        predictions[["true_labels", "predicted_labels"]] = predictions[["true_labels", "predicted_labels"]].replace(
            labels
        )
        labels = list(labels.values())

    if not conf_matrix_kwargs.get("cmap", None):
        # check if fau_r is registered as colormap
        if "fau_r" not in plt.colormaps():
            _register_fau_r()
        conf_matrix_kwargs["cmap"] = "fau_r"
    conf_matrix_kwargs.setdefault("colorbar", False)

    ConfusionMatrixDisplay.from_predictions(
        predictions["true_labels"],
        predictions["predicted_labels"],
        labels=labels,
        ax=ax,
        **conf_matrix_kwargs,
    )
    if kwargs.get("despine", True):
        for spine in ["top", "bottom", "left", "right"]:
            ax.spines[spine].set_color("None")

    ax.set_ylabel(f"True {label_name}")
    ax.set_xlabel(f"Predicted {label_name}")

    return fig, ax


def plot_conf_matrix_proba(
    predictions: pd.DataFrame,
    labels: Sequence[str],
    label_col: Optional[str] = "label",
    label_name: Optional[str] = "label",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot confusion matrix from prediction probabilities.

    Parameters
    ----------
    predictions : :class:`~pandas.DataFrame`
        dataframe with predictions as probabilities
    labels : list
        list of labels
    label_col : str, optional
        name of the label column in the input dataframe. Default: ``"label"``
    label_name : str, optional
        name of the 'label' in the axis titles. Default: "label" to yield "True label" and "Predicted label"
    **kwargs
        additional keyword arguments to pass to :func:`plt.subplots`

    """
    # check if ax is given
    ax = kwargs.get("ax")
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = ax.get_figure()

    # check if fau_r is registered as colormap
    if "fau_r" not in plt.colormaps():
        _register_fau_r()

    conf_matrix_proba = _conf_matrix_from_proba_df(predictions, label_col=label_col, label_order=labels)

    sns.heatmap(conf_matrix_proba, cmap="fau_r", annot=True, cbar=False, square=True, ax=ax)
    ax.set_ylabel(f"True {label_name}")
    ax.set_xlabel(f"Predicted {label_name}")

    return fig, ax


def metric_summary_to_latex(
    permuter_or_df: Union[SklearnPipelinePermuter, pd.DataFrame],
    metrics: Optional[Sequence[str]] = None,
    pipeline_steps: Optional[Sequence[str]] = None,
    si_table_format: Optional[str] = None,
    highlight_best: Optional[str] = None,
    **kwargs,
) -> str:
    """Return a latex table with the performance metrics of the pipeline combinations.

    Notes
    -----
    This method is a legacy method that is kept for backwards compatibility with older pickled instances of the
    :class:`SklearnPipelinePermuter` class. It is recommended to use the :meth:`SklearnPipelinePermuter.metric_summary`
    method instead.

    See Also
    --------
    :meth:`SklearnPipelinePermuter.metric_summary_to_latex`


    Parameters
    ----------
    permuter_or_df : :class:`SklearnPipelinePermuter` or :class:`~pandas.DataFrame`
        :class:`SklearnPipelinePermuter` instance or dataframe with performance metrics.
    metrics : list of str, optional
        list of metrics to include in the table or ``None`` to use all available metrics in the dataframe.
        Default: ``None``
    pipeline_steps : list of str, optional
        list of pipeline steps to include in the table index or ``None`` to show all available pipeline steps
        as table index. Default: ``None``
    si_table_format : str, optional
        table format for the ``siunitx`` package or ``None`` to use the default format. Default: ``None``
    highlight_best : bool or str, optional
        Whether to highlight the pipeline with the best value in each column or not.
        *  If ``highlight_best`` is a boolean, the best pipeline is highlighted in each column.
        *  If ``highlight_best`` is a string, the best pipeline is highlighted in the column with the name
    **kwargs
        additional keyword arguments passed to :func:`~pandas.DataFrame.to_latex`

    """
    dummy_permuter = SklearnPipelinePermuter()
    if isinstance(permuter_or_df, SklearnPipelinePermuter):
        metric_summary = permuter_or_df.metric_summary
    else:
        metric_summary = permuter_or_df

    return dummy_permuter.metric_summary_to_latex(
        metric_summary,
        metrics=metrics,
        pipeline_steps=pipeline_steps,
        si_table_format=si_table_format,
        highlight_best=highlight_best,
        **kwargs,
    )
