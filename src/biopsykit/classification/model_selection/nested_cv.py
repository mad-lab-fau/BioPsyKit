"""Module with functions for model selection using "nested" cross-validation."""
import warnings
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import confusion_matrix, get_scorer
from sklearn.model_selection import BaseCrossValidator, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from biopsykit.classification.utils import split_train_test

__all__ = ["nested_cv_param_search"]


def nested_cv_param_search(  # pylint:disable=invalid-name # pylint:disable=too-many-branches
    X: np.ndarray,  # noqa
    y: np.ndarray,
    param_dict: Dict[str, Any],
    pipeline: Pipeline,
    outer_cv: BaseCrossValidator,
    inner_cv: BaseCrossValidator,
    groups: Optional[np.ndarray] = None,
    hyper_search_params: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """Perform a cross-validated parameter search with hyperparameter optimization within a outer cross-validation.

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
        Pipeline of sklearn transforms and estimators to perform hyperparameter search with.
    outer_cv : `CV splitter <https://scikit-learn.org/stable/glossary.html#term-CV-splitter>`_
        Cross-validation object determining the cross-validation splitting strategy of the outer cross-validation.
    inner_cv : `CV splitter`_
        Cross-validation object determining the cross-validation splitting strategy of the hyperparameter search.
    groups : array-like of shape (`n_samples`,)
        Group labels for the samples used while splitting the dataset into train/test set. Only used in conjunction
        with a "Group"``cv`` instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).
        Default: ``None``
    hyper_search_params : dict, optional
        Dictionary specifying which hyperparameter search method to use (or ``None`` to use grid-search).

            * "grid" (:class:`~sklearn.model_selection.GridSearchCV`): To perform a grid-search pass a dict in the form
              of ``{"search_method": "grid"}``.
            * "random" (:class:`~sklearn.model_selection.RandomizedSearchCV`): To perform a randomized-search pass a
              dict in the form of ``{"search_method": "random", "n_iter": xx}``, where ``"n_iter"`` corresponds to the
              number of parameter settings that are sampled.

    kwargs : Additional arguments to be passed to the hyperparameter search class instance
             (e.g., :class:`~sklearn.model_selection.GridSearchCV` or
             :class:`~sklearn.model_selection.RandomizedSearchCV`).


    Returns
    -------
    dict
        Dictionary with hyperparameter search results. The result dictionary has the following entries:

        - "param_search": list with hyperparameter search class instances
          (e.g., :class:`~sklearn.model_selection.GridSearchCV`) used for hyperparameter search for each outer fold
          (determined by ``outer_cv``).
        - "test_score":  list with test scores of the best estimator on the respective test set for each outer fold.
        - "cv_results": list of ``cv_results_`` attributes of hyperparameter search class
          (e.g., :class:`~sklearn.model_selection.GridSearchCV`).
          Each entry of "cv_results" is a results dictionary of the respective fold with keys as column headers and
          values as columns, that can be imported into a pandas DataFrame.
        - "best_estimator" list of ``best_estimator_`` attributes of hyperparameter search class
          (e.g., :class:`~sklearn.model_selection.GridSearchCV`). Each entry of "best_estimator" is the estimator that
          was chosen by the hyperparameter in the respective fold, i.e. the estimator which gave the highest
          average score (or smallest loss if specified) on the test data.
        - "conf_matrix": list of confusion matrices from test scores for each outer fold

    See Also
    --------
    :class:`~sklearn.model_selection.GridSearchCV`
        sklearn grid-search
    :class:`~sklearn.model_selection.RandomizedSearchCV`
        sklearn randomized-search

    """
    if hyper_search_params is None:
        hyper_search_params = {"search_method": "grid"}

    scoring = kwargs.pop("scoring", None)
    kwargs, scoring_dict = _setup_scoring_dict(scoring, **kwargs)

    cols = [
        "param_search",
        "cv_results",
        "best_estimator",
        "conf_matrix",
        "predicted_labels",
        "true_labels",
        "train_indices",
        "test_indices",
    ]
    for scorer in scoring_dict:
        cols.append(f"test_{scorer}")
    results_dict = {key: [] for key in cols}

    for train, test in tqdm(list(outer_cv.split(X, y, groups)), desc="Outer CV"):
        cv_obj = _get_param_search_cv_object(
            pipeline, param_dict, inner_cv, scoring_dict, hyper_search_params, **kwargs
        )

        if groups is None:
            x_train, x_test, y_train, y_test = split_train_test(X, y, train, test)
            groups_train = None
        else:
            (  # pylint:disable=unbalanced-tuple-unpacking
                x_train,
                x_test,
                y_train,
                y_test,
                groups_train,
                _,
            ) = split_train_test(X, y, train, test, groups)

        cv_obj = _fit_cv_obj_one_fold(cv_obj, x_train, y_train, groups_train)

        results_dict["param_search"].append(cv_obj)
        for scorer in scoring_dict:
            results_dict["test_{}".format(scorer)].append(
                get_scorer(scorer)._score_func(y_test, cv_obj.predict(x_test))
            )
        results_dict["train_indices"].append(train)
        results_dict["test_indices"].append(test)
        results_dict["predicted_labels"].append(cv_obj.predict(x_test))
        results_dict["true_labels"].append(y_test)
        results_dict["cv_results"].append(cv_obj.cv_results_)
        results_dict["best_estimator"].append(cv_obj.best_estimator_)
        try:
            results_dict["conf_matrix"].append(confusion_matrix(y_test, cv_obj.predict(x_test), normalize=None))
        except ValueError as e:
            if "Classification metrics can't handle a mix of multiclass and continuous targets" in e.args[0]:
                warnings.warn("Cannot compute confusion matrix for regression tasks.")

    return results_dict


def _setup_scoring_dict(scoring, **kwargs):
    scoring_dict = {}
    if scoring is not None:
        if isinstance(scoring, str):
            kwargs["refit"] = scoring
            scoring = [scoring]

        for score in scoring:
            scoring_dict.setdefault(score, score)
    return kwargs, scoring_dict


def _fit_cv_obj_one_fold(cv_obj, x_train, y_train, groups_train):
    try:
        if groups_train is None:
            cv_obj.fit(x_train, y_train)
        else:
            cv_obj.fit(x_train, y_train, groups=groups_train)
    except ValueError as e:
        if "Classification metrics can't handle a mix of multiclass and continuous targets" in e.args[0]:
            raise ValueError(
                "Error when attempting to fit estimator. "
                "It seems that you are trying to fit a regression model, "
                "but specified metrics for classification. "
                "Please check your code and provide other evaluation metrics if necessary!"
            ) from e
        if "An empty dict was passed." in e.args[0]:
            raise ValueError("No scoring metric was specified for the estimator!") from e
        raise e
    return cv_obj


def _get_param_search_cv_object(
    pipeline: Pipeline,
    param_dict: Dict[str, Any],
    inner_cv: BaseCrossValidator,
    scoring_dict: Dict[str, str],
    hyper_search_config: Dict[str, Any],
    **kwargs,
):
    random_state = kwargs.pop("random_state", None)
    if hyper_search_config["search_method"] == "random":
        return RandomizedSearchCV(
            pipeline,
            param_distributions=param_dict,
            cv=inner_cv,
            scoring=scoring_dict,
            n_iter=hyper_search_config["n_iter"],
            random_state=random_state,
            **kwargs,
        )
    if hyper_search_config["search_method"] == "grid":
        return GridSearchCV(pipeline, param_grid=param_dict, cv=inner_cv, scoring=scoring_dict, **kwargs)
    raise ValueError("Unknown search method {}".format(hyper_search_config["search_method"]))
