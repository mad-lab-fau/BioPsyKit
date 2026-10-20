"""Regression tests for scorer semantics and ownership of nested-CV inputs."""

from copy import deepcopy

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, fbeta_score, get_scorer, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from biopsykit.classification.model_selection import nested_cv_param_search


@pytest.fixture
def classification_data():
    return make_classification(
        n_samples=60, n_features=4, n_informative=3, n_redundant=0, class_sep=0.5, random_state=42
    )


def _run(data, scoring, *, pipeline=None, **kwargs):
    if pipeline is None:
        pipeline = Pipeline([("model", LogisticRegression(random_state=3))])
    return nested_cv_param_search(
        *data,
        {},
        pipeline,
        StratifiedKFold(3),
        StratifiedKFold(2),
        scoring=scoring,
        **kwargs,
    )


def _assert_scores(result, data, scorers):
    """Compare every outer score with a public scorer evaluated on that fold's fitted estimator."""
    x, y = data
    for fold, (estimator, test) in enumerate(zip(result["best_estimator"], result["test_indices"])):
        for name, scorer in scorers.items():
            assert result[f"test_{name}"][fold] == pytest.approx(scorer(estimator, x[test], y[test]))


def test_negative_loss_retains_its_sign_in_the_original_audit_example():
    x = np.arange(20).reshape(-1, 1)
    y = np.arange(20, dtype=float)

    result = nested_cv_param_search(
        x, y, {}, Pipeline([("model", DummyRegressor())]), KFold(2), KFold(2), scoring="neg_mean_absolute_error"
    )

    assert result["test_neg_mean_absolute_error"] == pytest.approx([-10.0, -10.0])
    _assert_scores(result, (x, y), {"neg_mean_absolute_error": get_scorer("neg_mean_absolute_error")})


@pytest.mark.parametrize("scoring", ["accuracy", "neg_log_loss", "roc_auc"])
def test_named_scorers_use_their_public_estimator_interface(classification_data, scoring):
    result = _run(classification_data, scoring)

    _assert_scores(result, classification_data, {scoring: get_scorer(scoring)})
    if scoring == "roc_auc":
        x, y = classification_data
        label_scores = [
            roc_auc_score(y[test], estimator.predict(x[test]))
            for estimator, test in zip(result["best_estimator"], result["test_indices"])
        ]
        assert not np.allclose(result["test_roc_auc"], label_scores)


def test_auc_works_with_a_decision_function_without_predict_proba(classification_data):
    result = _run(classification_data, "roc_auc", pipeline=Pipeline([("model", SVC())]))

    _assert_scores(result, classification_data, {"roc_auc": get_scorer("roc_auc")})


def test_multiclass_scorer_keeps_its_configured_averaging():
    data = make_classification(n_samples=60, n_features=5, n_informative=4, n_redundant=0, n_classes=3, random_state=7)
    result = _run(data, "f1_macro")

    _assert_scores(result, data, {"f1_macro": get_scorer("f1_macro")})


def test_dictionary_preserves_aliases_and_scorer_arguments(classification_data):
    scoring = {"f2": make_scorer(fbeta_score, beta=2, average="macro"), "loss": "neg_log_loss"}
    result = _run(classification_data, scoring, refit="f2")

    _assert_scores(result, classification_data, {"f2": scoring["f2"], "loss": get_scorer("neg_log_loss")})
    assert scoring["loss"] == "neg_log_loss"


@pytest.mark.parametrize("container", [list, tuple])
def test_multiple_named_scorers_are_all_reported(classification_data, container):
    scoring = container(["accuracy", "neg_log_loss"])
    result = _run(classification_data, scoring, refit="neg_log_loss")

    _assert_scores(result, classification_data, {name: get_scorer(name) for name in scoring})


@pytest.mark.parametrize("scoring", [make_scorer(fbeta_score, beta=2), None])
def test_single_callable_and_default_estimator_scoring(classification_data, scoring):
    result = _run(classification_data, scoring)
    expected = scoring if scoring is not None else lambda estimator, x, y: estimator.score(x, y)

    _assert_scores(result, classification_data, {"score": expected})


def _multiple_scores(estimator, x, y):
    return {"accuracy": accuracy_score(y, estimator.predict(x)), "loss": get_scorer("neg_log_loss")(estimator, x, y)}


def test_callable_returning_multiple_scores(classification_data):
    result = _run(classification_data, _multiple_scores, refit="loss")

    _assert_scores(
        result, classification_data, {"accuracy": get_scorer("accuracy"), "loss": get_scorer("neg_log_loss")}
    )


def _refit_first_candidate(cv_results):
    return 0


@pytest.mark.parametrize("refit", [True, "accuracy", _refit_first_candidate])
def test_explicit_refit_strategy_is_respected(classification_data, refit):
    result = _run(classification_data, "accuracy", refit=refit)

    expected = "accuracy" if refit is True else refit
    assert all(search.refit == expected for search in result["param_search"])
    _assert_scores(result, classification_data, {"accuracy": get_scorer("accuracy")})


def test_refit_false_is_rejected_before_fitting(classification_data):
    with pytest.raises(ValueError, match="refit"):
        _run(classification_data, "accuracy", refit=False)


@pytest.mark.parametrize("seed_kwargs", [{}, {"random_state": None}, {"random_state": 99}])
def test_splitter_seeds_are_preserved_unless_explicitly_overridden(classification_data, seed_kwargs):
    x, y = classification_data
    outer = StratifiedKFold(3, shuffle=True, random_state=17)
    inner = StratifiedKFold(2, shuffle=True, random_state=23)
    expected_outer = deepcopy(outer)
    if seed_kwargs.get("random_state") is not None:
        expected_outer.random_state = seed_kwargs["random_state"]
    expected_inner_seed = seed_kwargs.get("random_state") or 23

    result = nested_cv_param_search(
        x, y, {}, Pipeline([("model", LogisticRegression())]), outer, inner, scoring="accuracy", **seed_kwargs
    )

    assert outer.random_state == 17
    assert inner.random_state == 23
    for (train, test), actual_train, actual_test in zip(
        expected_outer.split(x, y), result["train_indices"], result["test_indices"]
    ):
        np.testing.assert_array_equal(actual_train, train)
        np.testing.assert_array_equal(actual_test, test)
    assert all(search.cv.random_state == expected_inner_seed for search in result["param_search"])
    assert all(search.cv is not inner for search in result["param_search"])


def test_splitters_with_random_state_objects_do_not_advance_caller_state(classification_data):
    x, y = classification_data
    outer = StratifiedKFold(3, shuffle=True, random_state=np.random.RandomState(17))
    inner = StratifiedKFold(2, shuffle=True, random_state=np.random.RandomState(23))
    states = [deepcopy(splitter.random_state.get_state()) for splitter in (outer, inner)]

    nested_cv_param_search(x, y, {}, Pipeline([("model", LogisticRegression())]), outer, inner, scoring="accuracy")

    for splitter, before in zip((outer, inner), states):
        after = splitter.random_state.get_state()
        assert before[0] == after[0]
        np.testing.assert_array_equal(before[1], after[1])
        assert before[2:] == after[2:]


def test_randomized_search_and_groups_keep_the_same_outer_scoring_contract(classification_data):
    x, y = classification_data
    groups = np.repeat(np.arange(12), 5)
    result = nested_cv_param_search(
        x,
        y,
        {"model__C": [0.1, 1.0]},
        Pipeline([("model", LogisticRegression())]),
        GroupKFold(3),
        GroupKFold(2),
        groups=groups,
        hyper_search_params={"search_method": "random", "n_iter": 2},
        scoring={"loss": "neg_log_loss"},
        refit="loss",
        random_state=11,
    )

    _assert_scores(result, classification_data, {"loss": get_scorer("neg_log_loss")})
    for search, train, test in zip(result["param_search"], result["train_indices"], result["test_indices"]):
        assert isinstance(search, RandomizedSearchCV)
        assert search.random_state == 11
        assert set(groups[train]).isdisjoint(groups[test])


def test_nested_search_matches_sklearn_cross_validate_reference(classification_data):
    """Compare full nested optimization and outer scores against sklearn's own composition."""
    x, y = classification_data
    pipeline = Pipeline([("model", LogisticRegression(random_state=3))])
    grid = {"model__C": [0.01, 0.1, 1.0, 10.0]}
    outer = StratifiedKFold(3, shuffle=True, random_state=17)
    inner = StratifiedKFold(2, shuffle=True, random_state=23)
    scorers = {"accuracy": "accuracy", "loss": "neg_log_loss"}
    reference = cross_validate(
        GridSearchCV(pipeline, grid, cv=inner, scoring=scorers, refit="loss"),
        x,
        y,
        cv=outer,
        scoring=scorers,
        return_estimator=True,
    )

    result = nested_cv_param_search(x, y, grid, pipeline, outer, inner, scoring=scorers, refit="loss")

    for name in scorers:
        np.testing.assert_allclose(result[f"test_{name}"], reference[f"test_{name}"])
    for actual, expected in zip(result["param_search"], reference["estimator"]):
        assert actual.best_params_ == expected.best_params_
        assert actual.best_score_ == pytest.approx(expected.best_score_)
