"""Module with functions for model selection using "nested" cross-validation."""
from typing import Dict, Any, Optional

import numpy as np

from sklearn.metrics import confusion_matrix, get_scorer
from sklearn.model_selection import BaseCrossValidator, GridSearchCV
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from biopsykit.classification.utils import split_train_test

__all__ = ["nested_cv_grid_search"]


def nested_cv_grid_search(  # pylint:disable=invalid-name
    X: np.ndarray,  # noqa
    y: np.ndarray,
    param_dict: Dict[str, Any],
    pipeline: Pipeline,
    outer_cv: BaseCrossValidator,
    inner_cv: BaseCrossValidator,
    groups: Optional[np.ndarray] = None,
    **kwargs,
):
    """Perform a cross-validated grid-search with hyperparameter optimization within a outer cross-validation.

    Parameters
    ----------
    X : array-like of shape (`n_samples`, `n_features`)
        Training vector, where `n_samples` is the number of samples and `n_features` is the number of features.
    y : array-like of shape (`n_samples`, `n_output`) or (`n_samples`,)
        Target (i.e., class labels) relative to X for classification or regression.
    param_dict : dict or list of dicts
        Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values,
        or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are explored.
        This enables searching over any sequence of parameter settings.
    pipeline : :class:`~sklearn.pipeline.Pipeline`
        Pipeline of sklearn transforms and estimators to perform grid-search on.
    outer_cv : `CV splitter <https://scikit-learn.org/stable/glossary.html#term-CV-splitter>`_
        Cross-validation object determining the cross-validation splitting strategy of the outer cross-validation.
    inner_cv : `CV splitter`_
        Cross-validation object determining the cross-validation splitting strategy of the grid-search.
    groups : array-like of shape (`n_samples`,)
        Group labels for the samples used while splitting the dataset into train/test set. Only used in conjunction
        with a "Group"``cv`` instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).
        Default: ``None``
    kwargs : Additional arguments to be passed to the :class:`~sklearn.model_selection.GridSearchCV` instance.


    Returns
    -------
    dict
        Dictionary with grid search results. The result dictionary has the following entries:

        - "grid_search": list with :class:`~sklearn.model_selection.GridSearchCV` instances used for grid search for
          each outer fold (determined by ``outer_cv``).
        - "test_score":  list with test scores of the best estimator on the respective test set for each outer fold.
        - "cv_results": list of ``cv_results_`` attributes of :class:`~sklearn.model_selection.GridSearchCV`.
          Each entry of "cv_results" is a results dictionary of the respective fold with keys as column headers and
          values as columns, that can be imported into a pandas DataFrame.
        - "best_estimator" list of ``best_estimator_`` attributes of :class:`~sklearn.model_selection.GridSearchCV`.
          Each entry of "best_estimator" is the estimator that was chosen by the grid-search in the respective fold,
          i.e. the estimator which gave the highest average score (or smallest loss if specified) on the test data.
        - "conf_matrix": list of confusion matrices from test scores for each outer fold

    See Also
    --------
    :class:`~sklearn.model_selection.GridSearchCV`
        sklearn grid-search

    """
    scoring_dict = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"}
    scoring = kwargs.pop("scoring")
    scoring_dict.setdefault(scoring, scoring)
    kwargs["refit"] = scoring

    cols = ["grid_search", "cv_results", "best_estimator", "conf_matrix", "predicted_labels", "true_labels"]
    for scorer in scoring_dict:
        cols.append("test_{}".format(scorer))
    results_dict = {key: [] for key in cols}

    for train, test in tqdm(list(outer_cv.split(X, y, groups)), desc="Outer CV"):
        if groups is None:
            x_train, x_test, y_train, y_test = split_train_test(X, y, train, test)
            grid = GridSearchCV(pipeline, param_grid=param_dict, cv=inner_cv, scoring=scoring_dict, **kwargs)
            grid.fit(x_train, y_train)
        else:
            (  # pylint:disable=unbalanced-tuple-unpacking
                x_train,
                x_test,
                y_train,
                y_test,
                groups_train,
                _,
            ) = split_train_test(X, y, train, test, groups)
            grid = GridSearchCV(pipeline, param_grid=param_dict, cv=inner_cv, scoring=scoring_dict, **kwargs)
            grid.fit(x_train, y_train, groups=groups_train)

        results_dict["grid_search"].append(grid)
        results_dict["test_{}".format(scoring)].append(grid.score(x_test, y_test))
        for scorer in scoring_dict:
            if scorer == scoring:
                continue
            results_dict["test_{}".format(scorer)].append(get_scorer(scorer)._score_func(y_test, grid.predict(x_test)))
        results_dict["predicted_labels"].append(grid.predict(x_test))
        results_dict["true_labels"].append(y_test)
        results_dict["cv_results"].append(grid.cv_results_)
        results_dict["best_estimator"].append(grid.best_estimator_)
        results_dict["conf_matrix"].append(confusion_matrix(y_test, grid.predict(x_test), normalize=None))

    return results_dict
