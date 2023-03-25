"""Functions to analyze classification results."""
from biopsykit.classification.analysis._analysis import (
    metric_summary_to_latex,
    plot_conf_matrix,
    plot_conf_matrix_proba,
    predict_proba_from_estimator,
    predictions_as_df,
)

__all__ = [
    "predictions_as_df",
    "predict_proba_from_estimator",
    "plot_conf_matrix",
    "plot_conf_matrix_proba",
    "metric_summary_to_latex",
]
