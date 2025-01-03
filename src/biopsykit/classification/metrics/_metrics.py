"""Metrics for classification tasks."""
from inspect import getmembers

import numpy as np
import pandas as pd
import sklearn.metrics

from biopsykit.utils._types import str_t


def _apply_score(row: pd.Series, score_func, pos_label: str):
    true_labels_folds = row[0]
    predicted_labels_folds = row[1]
    scores = [
        score_func(true_labels, predicted_labels, pos_label=pos_label)
        for true_labels, predicted_labels in zip(true_labels_folds, predicted_labels_folds)
    ]
    return pd.Series(scores)


def compute_additional_metrics(metric_summary: pd.DataFrame, metrics: str_t, pos_label: str):
    """Compute additional classification metrics from a ``SklearnPipelinePermuter`` metric summary dataframe.

    Parameters
    ----------
    metric_summary : :class:`~pandas.DataFrame`
        metric summary dataframe from :class:`~biopsykit.classification.model_selection.SklearnPipelinePermuter`
    metrics : list
        list of additional metrics to compute
    pos_label : str
        positive label

    """
    metric_slice = metric_summary[["true_labels_folds", "predicted_labels_folds"]].copy()
    metric_out = {}

    # ensure list
    if isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        score_funcs = dict(getmembers(sklearn.metrics))
        if metric in score_funcs:
            score_func = score_funcs[f"{metric}"]
        elif f"{metric}_score" in score_funcs:
            score_func = score_funcs[f"{metric}_score"]
            metric = f"{metric}_score"  # noqa: PLW2901
        else:
            raise ValueError(f"Metric '{metric}' not found.")
        metric_out[metric] = metric_slice.apply(_apply_score, args=(score_func, pos_label), axis=1)
    metric_out = pd.concat(metric_out, names=["score", "folds"], axis=1)

    metric_out = metric_out.stack(["score", "folds"])
    metric_out = metric_out.groupby(metric_out.index.names[:-1]).agg(
        [("mean", lambda x: np.mean), ("std", lambda x: np.std(x))]  # noqa: ARG005
    )

    metric_out = metric_out.unstack("score").sort_index(axis=1, level="score")
    metric_out.columns = metric_out.columns.map("_test_".join)
    metric_summary = metric_summary.join(metric_out)

    return metric_summary
