"""Module for systematically evaluating different combinations of sklearn pipelines."""
import functools
import pickle
import re
from collections.abc import Sequence
from copy import deepcopy
from inspect import getmembers, signature
from itertools import product
from pathlib import Path
from shutil import rmtree
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import sklearn.metrics
from joblib import Memory
from numpy.random import RandomState
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm
from typing_extensions import Self

from biopsykit.classification.model_selection import nested_cv_param_search
from biopsykit.classification.utils import _PipelineWrapper, merge_nested_dicts
from biopsykit.utils._datatype_validation_helper import _assert_file_extension
from biopsykit.utils._types import path_t, str_t

__all__ = ["SklearnPipelinePermuter"]

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
    "SelectFromModel": "SFM",
    "GaussianNB": "NB",
    "KNeighborsClassifier": "kNN",
    "DecisionTreeClassifier": "DT",
    "SVC": "SVM",
    "RandomForestClassifier": "RF",
    "MLPClassifier": "MLP",
    "AdaBoostClassifier": "Ada",
}


class SklearnPipelinePermuter:
    """Class for systematically evaluating different sklearn pipeline combinations."""

    def __init__(
        self,
        model_dict: Optional[dict[str, dict[str, BaseEstimator]]] = None,
        param_dict: Optional[dict[str, Optional[Union[Sequence[dict[str, Any]], dict[str, Any]]]]] = None,
        hyper_search_dict: Optional[dict[str, dict[str, Any]]] = None,
        random_state: Optional[int] = None,
    ):
        """Class for systematically evaluating different sklearn pipeline combinations.

        This class can be used to, for instance, evaluate combinations of different feature selection methods
        (e.g., :class:`~sklearn.feature_selection.SelectKBest`,
        :class:`~sklearn.feature_selection.SequentialFeatureSelector`) with different estimators
        (e.g., :class:`~sklearn.svm.SVC`, :class:`~sklearn.tree.DecisionTreeClassifier`), any much more.

        For all combinations, hyperparameter search (e.g., using grid-search or randomized-search) can be performed by
        passing one joint parameter grid (see Examples).

        Parameters
        ----------
        model_dict : dict
            Dictionary specifying the different transformers and estimators to evaluate.
            Each pipeline step corresponds to one dictionary entry and has the name of the pipeline step (str) as key.
            The values are again dictionaries with the transformer/estimator names as keys and instances of the
            transformers/estimators as values
        param_dict : dict
            Nested dictionary specifying the parameter settings to try per transformer/estimator. The dictionary has
            the transformer/estimator names (str) as keys and parameter dictionaries as values. Each parameter
            dictionary has parameters names (str) as keys and lists of parameter settings to try as values, or a list
            of such dictionaries, in which case the grids spanned by each dictionary in the list are explored.
            This enables searching over any sequence of parameter settings.
        hyper_search_dict : dict, optional
            Nested dictionary specifying the method for hyperparameter search (e.g., whether to use "grid" for
            grid-search or "random" for randomized-search) for each estimator. By default, "grid-search" is used
            for each estimator unless individually specified otherwise.
        random_state : int, optional
            Controls the random seed passed to each estimator and each splitter. By default, no random seed is passed.
            Set this to an integer for reproducible results across multiple program calls.

        Examples
        --------
        >>> from sklearn import datasets
        >>> from sklearn.preprocessing import StandardScaler, MinMaxScaler
        >>> from sklearn.feature_selection import SelectKBest, RFE
        >>> from sklearn.neighbors import KNeighborsClassifier
        >>> from sklearn.svm import SVC
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.ensemble import AdaBoostClassifier
        >>> from sklearn.model_selection import KFold
        >>>
        >>> from biopsykit.classification.model_selection import SklearnPipelinePermuter
        >>>
        >>> breast_cancer = datasets.load_breast_cancer()
        >>> X = breast_cancer.data
        >>> y = breast_cancer.target
        >>>
        >>> model_dict = {
        >>>    "scaler": {
        >>>         "StandardScaler": StandardScaler(),
        >>>         "MinMaxScaler": MinMaxScaler(),
        >>>     },
        >>>     "reduce_dim": {
        >>>         "SelectKBest": SelectKBest(),
        >>>         "RFE": RFE(SVC(kernel="linear", C=1))
        >>>     },
        >>>     "clf" : {
        >>>         "KNeighborsClassifier": KNeighborsClassifier(),
        >>>         "DecisionTreeClassifier": DecisionTreeClassifier(),
        >>>         "SVC": SVC(),
        >>>         "AdaBoostClassifier": AdaBoostClassifier(),
        >>>     }
        >>> }
        >>>
        >>> param_dict = {
        >>>     "StandardScaler": None,
        >>>     "MinMaxScaler": None,
        >>>     "SelectKBest": { "k": [2, 4, 6, 8, "all"] },
        >>>     "RFE": { "n_features_to_select": [2, 4, 6, 8, None] },
        >>>     "KNeighborsClassifier": { "n_neighbors": [2, 4, 6, 8], "weights": ["uniform", "distance"] },
        >>>     "DecisionTreeClassifier": {"criterion": ['gini', 'entropy'], "max_depth": [2, 4, 6, 8, 10] },
        >>>     "AdaBoostClassifier": {
        >>>         "base_estimator": [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)],
        >>>         "n_estimators": np.arange(20, 210, 10),
        >>>         "learning_rate": np.arange(0.6, 1.1, 0.1)
        >>>     },
        >>>     "SVC": [
        >>>         {
        >>>             "kernel": ["linear"],
        >>>             "C": np.logspace(start=-3, stop=3, num=7)
        >>>         },
        >>>         {
        >>>             "kernel": ["rbf"],
        >>>             "C": np.logspace(start=-3, stop=3, num=7),
        >>>             "gamma": np.logspace(start=-3, stop=3, num=7)
        >>>         }
        >>>     ]
        >>> }
        >>>
        >>> # AdaBoost hyperparameters should be optimized using randomized-search, all others using grid-search
        >>> hyper_search_dict = {
        >>>     "AdaBoostClassifier": {"search_method": "random", "n_iter": 30}
        >>> }
        >>>
        >>> pipeline_permuter = SklearnPipelinePermuter(model_dict, param_dict, hyper_search_dict)
        >>> pipeline_permuter.fit(X, y, outer_cv=KFold(), inner_cv=KFold())

        """
        self.models: dict[str, dict[str, BaseEstimator]] = {}
        """Dictionary with pipeline steps and the different transformers/estimators per step."""

        self.params: dict[str, Optional[Union[Sequence[dict[str, Any]], dict[str, Any]]]] = {}
        """Dictionary with parameter sets to test for the different transformers/estimators per pipeline step."""

        self.model_combinations: Sequence[tuple[tuple[str, str], ...]] = []
        """List of model combinations, i.e. permutations of the different transformers/estimators for
        each pipeline step."""

        self.hyper_search_dict: dict[str, dict[str, Any]] = {}
        """Dictionary specifying the selected hyperparameter search method for each estimator."""

        self.param_searches: dict[tuple[str, str], dict[str, Any]] = {}
        """Dictionary with parameter search results for each pipeline step combination."""

        self.results: Optional[pd.DataFrame] = None
        """Dataframe with parameter search results of each pipeline step combination."""

        self.scoring: str_t = ""
        """Scoring used as metric for optimization during hyperparameter search."""

        self.refit: str = ""

        self.random_state: Optional[RandomState] = None

        self._results_set: bool = False

        if model_dict is None and param_dict is None:
            # create empty instance
            return

        self.random_state = RandomState(random_state)

        self._set_permuter_params(model_dict, param_dict, hyper_search_dict)

    def _set_permuter_params(self, model_dict, param_dict, hyper_search_dict):
        self._check_missing_params(model_dict, param_dict)

        if hyper_search_dict is None:
            hyper_search_dict = {}
        self.hyper_search_dict = hyper_search_dict.copy()

        clf_list = model_dict[list(model_dict.keys())[-1]]
        for clf in clf_list:
            # fill the dict with the default search method (grid-search) for the classifiers that are not
            # specified explicitly
            self.hyper_search_dict.setdefault(clf, {"search_method": "grid"})

        model_combinations = list(product(*[[(step, k) for k in list(model_dict[step].keys())] for step in model_dict]))

        # assert that all entries of the param dict are lists for uniform handling
        for k, v in param_dict.items():
            if isinstance(v, dict):
                param_dict[k] = [v]

        model_dict = deepcopy(model_dict)
        self.models = self._initialize_models(model_dict)
        self.params = param_dict
        self.model_combinations = model_combinations

    def _initialize_models(self, model_dict: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if self.random_state is None:
            return model_dict
        for _k, v in model_dict.items():
            # add fixed random state to each estimator if it has a random_state parameter
            for estimator in v.values():
                if hasattr(estimator, "random_state"):
                    estimator.random_state = self.random_state
        return model_dict

    @property
    def results(self):
        """Parameter search results of each pipeline step combination.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Dataframe with parameter search results of each pipeline step combination

        """
        if self._results is None:
            self._results = self.pipeline_score_results()
        return self._results

    @results.setter
    def results(self, results):
        if results is None:
            self._results_set = False
        else:
            self._results_set = True
        self._results = results

    @classmethod
    def from_csv(cls, file_path: path_t, num_pipeline_steps: Optional[int] = 3) -> Self:
        """Create a new ``SklearnPipelinePermute`` instance from a csv file with exported results from parameter search.

        Parameters
        ----------
        file_path : :class:`pathlib.Path` or str
            path to csv file
        num_pipeline_steps : int
            integer specifying the number of steps in the pipeline. Used to infer pipeline steps from the
            :class:`~pandas.MultiIndex` in the dataframe. For instance, for a pipeline consisting of the steps
            "scaler", "reduce_dim", and "clf" pass "3" as ``num_pipeline_steps``

        Returns
        -------
        :class:`~biopsykit.classification.model_selection.SklearnPipelinePermuter`
            ``SklearnPipelinePermuter`` instance with results from csv file

        """
        # assert pathlib
        file_path = Path(file_path)
        _assert_file_extension(file_path, ".csv")
        score_summary = pd.read_csv(file_path)
        score_summary = score_summary.set_index(list(score_summary.columns)[: num_pipeline_steps + 2])
        pipeline_permuter = SklearnPipelinePermuter()
        pipeline_permuter.results = score_summary
        return pipeline_permuter

    def fit(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
        *,
        outer_cv: BaseCrossValidator,
        inner_cv: BaseCrossValidator,
        scoring: Optional[str_t] = None,
        use_cache: Optional[bool] = True,
        **kwargs,
    ):
        """Run fit for all pipeline combinations and sets of parameters.

        This function calls :func:`~biopsykit.classification.model_selection.nested_cv_param_search` for all
        Pipeline combinations and stores the results in the ``param_searches`` attribute.

        Parameters
        ----------
        X : array-like of shape (`n_samples`, `n_features`)
            Training vector, where `n_samples` is the number of samples and `n_features` is the number of features.
        y : array-like of shape (`n_samples`, `n_output`) or (`n_samples`,)
            Target (i.e., class labels) relative to X for classification or regression.
        outer_cv : `CV splitter`_
            Cross-validation object determining the cross-validation splitting strategy of the outer cross-validation.
        inner_cv : `CV splitter`_
            Cross-validation object determining the cross-validation splitting strategy of the hyperparameter search.
        scoring : str, optional
            A str specifying the scoring metric to use for evaluation.
        use_cache : bool, optional
            ``True`` to cache fitted transformer instances of the pipeline in a caching directory
            (can be provided by the additional parameter ``cachedir_name``), ``False`` otherwise. Default: ``True``
        **kwargs :
            Additional arguments that are passed to
            :func:`~biopsykit.classification.model_selection.nested_cv_parameter_search` and the hyperparameter search
            class instance (e.g., :class:`~sklearn.model_selection.GridSearchCV` or
            :class:`~sklearn.model_selection.RandomizedSearchCV`).

        """
        return self._fit(X=X, y=y, outer_cv=outer_cv, inner_cv=inner_cv, scoring=scoring, use_cache=use_cache, **kwargs)

    def _fit(  # noqa: PLR0912, C901
        self,
        *,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
        outer_cv: BaseCrossValidator,
        inner_cv: BaseCrossValidator,
        save_intermediate: Optional[bool] = False,
        file_path: Optional[path_t] = None,
        scoring: Optional[str_t] = None,
        use_cache: Optional[bool] = True,
        **kwargs,
    ):
        self.results = None

        if len(self.model_combinations) == 0:
            raise ValueError("No model combinations specified. Please specify at least one model combination.")

        kwargs.setdefault("n_jobs", -1)
        kwargs.setdefault("verbose", 1)
        kwargs.setdefault("error_score", "raise")

        # Create a temporary folder to store the transformers of the pipeline
        location = kwargs.pop("cachedir_name", "cachedir")
        memory = None
        if use_cache:
            memory = Memory(location=location, verbose=0)

        if scoring is None:
            scoring = "accuracy"
        self.scoring = scoring

        refit = kwargs.get("refit")
        if refit is None:
            refit = scoring
        self.refit = refit

        for model_combination in tqdm(self.model_combinations, desc="Pipeline Combinations"):
            if model_combination in self.param_searches:
                print(f"Skipping {model_combination} since this combination was already fitted!")
                # continue if we already tried this combination
                continue

            pipeline_params = [(m, self.params[k[1]]) for m, k in zip(self.models.keys(), model_combination)]
            pipeline_params = list(filter(lambda p: p[1] is not None, pipeline_params))
            pipeline_params = [(m, k_new) for m, k in pipeline_params for k_new in k if k is not None]

            cats = {p[0] for p in pipeline_params}
            pipeline_params = [list(filter(lambda p, c=cat: p[0] == c, pipeline_params)) for cat in cats]
            pipeline_params = list(product(*pipeline_params))

            pipeline_params = [
                tuple({f"{step[0]}__{k}": v for k, v in step[1].items()} for step in combi) for combi in pipeline_params
            ]
            pipeline_params = [{k: v for x in param for k, v in x.items()} for param in pipeline_params]

            if kwargs["verbose"] >= 1:
                print(
                    f"### Running hyperparameter search for pipeline: "
                    f"{model_combination} with {len(pipeline_params)} parameter grid(s):"
                )

            for j, param_dict in enumerate(pipeline_params):
                hyper_search_params = self.hyper_search_dict[model_combination[-1][1]]
                model_cls = [(step, self.models[step][m]) for step, m in model_combination]
                for i in range(len(model_cls)):
                    if isinstance(model_cls[i][1], BaseEstimator):
                        model_cls[i] = (model_cls[i][0], clone(model_cls[i][1]))

                pipeline = Pipeline(model_cls, memory=memory)
                if kwargs["verbose"] >= 1:
                    print(f"Parameter grid #{j} ({hyper_search_params}): {param_dict}")

                result_dict = nested_cv_param_search(
                    X,
                    y,
                    param_dict=param_dict,
                    pipeline=pipeline,
                    outer_cv=outer_cv,
                    inner_cv=inner_cv,
                    scoring=scoring,
                    hyper_search_params=hyper_search_params,
                    random_state=self.random_state,
                    **kwargs,
                )

                self.param_searches[model_combination] = result_dict
                if kwargs["verbose"] >= 1:
                    print("")

            if save_intermediate:
                # Save intermediate results to file
                self.to_pickle(file_path)
            if kwargs["verbose"] >= 1:
                print("")

        if use_cache:
            # Delete the temporary cache before exiting
            memory.clear(warn=False)
            rmtree(location)

    def fit_and_save_intermediate(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
        *,
        outer_cv: BaseCrossValidator,
        inner_cv: BaseCrossValidator,
        file_path: path_t,
        scoring: Optional[str_t] = None,
        use_cache: Optional[bool] = True,
        **kwargs,
    ):
        """Run fit for all pipeline combinations and sets of parameters and save intermediate results to file.

        This function calls :func:`~biopsykit.classification.model_selection.nested_cv_param_search` for all
        Pipeline combinations and stores the results in the ``param_searches`` attribute. After each model combination,
        the results are saved to a pickle file.

        Parameters
        ----------
        X : array-like of shape (`n_samples`, `n_features`)
            Training vector, where `n_samples` is the number of samples and `n_features` is the number of features.
        y : array-like of shape (`n_samples`, `n_output`) or (`n_samples`,)
            Target (i.e., class labels) relative to X for classification or regression.
        outer_cv : `CV splitter`_
            Cross-validation object determining the cross-validation splitting strategy of the outer cross-validation.
        inner_cv : `CV splitter`_
            Cross-validation object determining the cross-validation splitting strategy of the hyperparameter search.
        file_path : :class:`pathlib.Path` or str
            path to pickle file
        scoring : str, optional
            A str specifying the scoring metric to use for evaluation.
        use_cache : bool, optional
            ``True`` to cache fitted transformer instances of the pipeline in a caching directory
            (can be provided by the additional parameter ``cachedir_name``), ``False`` otherwise. Default: ``True``
        **kwargs :
            Additional arguments that are passed to
            :func:`~biopsykit.classification.model_selection.nested_cv_parameter_search` and the hyperparameter search
            class instance (e.g., :class:`~sklearn.model_selection.GridSearchCV` or
            :class:`~sklearn.model_selection.RandomizedSearchCV`).

        """
        return self._fit(
            X=X,
            y=y,
            outer_cv=outer_cv,
            inner_cv=inner_cv,
            save_intermediate=True,
            file_path=file_path,
            scoring=scoring,
            use_cache=use_cache,
            **kwargs,
        )

    @functools.lru_cache(maxsize=5)  # noqa: B019
    def pipeline_score_results(self) -> pd.DataFrame:
        """Return parameter search results for each pipeline combination.

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with parameter search results for each pipeline combination

        """
        if self._results_set:
            return self.results
        if len(self.param_searches) == 0:
            raise AttributeError(
                "No results available because pipelines were not fitted! Call `SklearnPipelinePermuter.fit()` first."
            )

        gs_param_list = []
        for param, gs in self.param_searches.items():
            dict_folds = {}
            for i, res in enumerate(gs["cv_results"]):
                df_res = pd.DataFrame(res)
                df_res = df_res.drop(columns=df_res.filter(like="time").columns)
                df_res.index.name = "parameter_combination_id"
                # note: the best_estimator from the fold does not necessarily correspond to the pipeline returned
                # from best_pipeline() because the best_pipeline is determined by averaging over all folds and
                # can hence have a different set of hyperparameters
                # best_estimator = gs["best_estimator"][i]
                # df_res["best_estimator"] = _PipelineWrapper(best_estimator)
                dict_folds[i] = df_res
            param_dict = {f"pipeline_{key}": val for key, val in param}
            df_gs = pd.concat(dict_folds, names=["outer_fold"])
            df_gs[list(param_dict.keys())] = [list(param_dict.values())] * len(df_gs)
            df_gs = df_gs.set_index(list(df_gs.filter(like="pipeline").columns), append=True)
            # reorder levels so that pipeline steps are the first index levels, then the hyperparameter
            # combination id, then the outer_fold id
            gs_param_list.append(df_gs)

        df_summary = pd.concat(gs_param_list)
        df_summary = df_summary.infer_objects()

        df_summary = df_summary.reorder_levels(
            df_summary.index.names[2:] + [df_summary.index.names[1]] + [df_summary.index.names[0]]
        )
        self.results = df_summary.sort_index().sort_index(axis=1)
        return self.results

    def metric_summary(
        self, additional_metrics: Optional[str_t] = None, pos_label: Optional[str] = None
    ) -> pd.DataFrame:
        """Return summary with all performance metrics for the `best-performing estimator` of each pipeline combination.

        The `best-performing estimator` for each pipeline combination is the `best_estimator_` that
        :class:`~sklearn.model_selection.GridSearchCV` returns for each outer fold, i.e. the pipeline which yielded
        the highest average test score (over all inner folds).

        Parameters
        ----------
        additional_metrics : str or list of str, optional
            additional metrics to compute. Default: ``None``. Available metrics can be found in scikit-learn's
            `metrics and scoring <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_
            module.
        pos_label : str, optional
            positive label for binary classification, must be specified if `additional_metrics` is specified.

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with performance metric summary the `best estimator` of each pipeline combination.

        """
        if len(self.param_searches) == 0:
            raise AttributeError(
                "No results available because pipelines were not fitted! Call `SklearnPipelinePermuter.fit()` first."
            )
        list_metric_summary = []
        for param_key, param_value in self.param_searches.items():
            param_dict = {f"pipeline_{key}": val for key, val in param_key}
            conf_matrix = np.sum(param_value["conf_matrix"], axis=0)
            true_labels = np.array(param_value["true_labels"], dtype="object")
            predicted_labels = np.array(param_value["predicted_labels"], dtype="object")
            train_indices = np.array(param_value["train_indices"], dtype="object")
            test_indices = np.array(param_value["test_indices"], dtype="object")
            df_metric = pd.DataFrame(param_dict, index=[0])

            df_metric["conf_matrix"] = [list(conf_matrix.flatten())]
            df_metric["conf_matrix_folds"] = [[cm.flatten() for cm in param_value["conf_matrix"]]]
            df_metric["true_labels"] = [np.concatenate(true_labels)]
            df_metric["true_labels_folds"] = [true_labels]
            df_metric["predicted_labels"] = [np.concatenate(predicted_labels)]
            df_metric["predicted_labels_folds"] = [predicted_labels]
            df_metric["train_indices"] = [np.concatenate(train_indices)]
            df_metric["train_indices_folds"] = [train_indices]
            df_metric["test_indices"] = [np.concatenate(test_indices)]
            df_metric["test_indices_folds"] = [test_indices]

            scoring = self.scoring
            if isinstance(scoring, str):
                scoring = [scoring]
            for score_key in scoring:
                key = f"test_{score_key}"
                test_scores = self.param_searches[param_key][key]
                df_metric[f"mean_{key}"] = np.mean(test_scores)
                df_metric[f"std_{key}"] = np.std(test_scores)
                df_metric[[f"{key}_fold_{i}" for i in range(len(test_scores))]] = list(test_scores)

            df_metric = df_metric.set_index(list(df_metric.columns)[: len(param_dict)])
            list_metric_summary.append(df_metric)

        metric_summary = pd.concat(list_metric_summary)

        if additional_metrics is not None:
            metric_summary = self.compute_additional_metrics(
                metric_summary, metrics=additional_metrics, pos_label=pos_label
            )

        return metric_summary

    def export_pipeline_score_results(self, file_path: path_t) -> None:
        """Export pipeline score results as csv file.

        Parameters
        ----------
        file_path : :class:`~pathlib.Path` or str
            file path to export

        """
        file_path = Path(file_path)
        _assert_file_extension(file_path, ".csv")
        self.results.to_csv(file_path)

    def export_metric_summary(self, file_path: path_t) -> None:
        """Export performance metric summary as csv file.

        Parameters
        ----------
        file_path : :class:`~pathlib.Path` or str
            file path to export

        """
        file_path = Path(file_path)
        _assert_file_extension(file_path, ".csv")
        self.metric_summary().to_csv(file_path, sep=";")

    def best_estimator_summary(self) -> pd.DataFrame:
        """Return a dataframe with the `best estimator` instances of all pipeline combinations for each fold.

        Each entry of the dataframe is a list of :class:`~sklearn.pipeline.Pipeline` objects whe returned the .

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with `best estimator` instances

        """
        best_estimator_list = []
        for param_key, param_value in self.param_searches.items():
            param_dict = {f"pipeline_{key}": val for key, val in param_key}
            df_be = pd.DataFrame(param_dict, index=[0])
            df_be["best_estimator"] = _PipelineWrapper(param_value["best_estimator"])
            df_be = df_be.set_index(list(df_be.columns)[:-1])
            best_estimator_list.append(df_be)

        return pd.concat(best_estimator_list)

    @functools.lru_cache(maxsize=5)  # noqa: B019
    def mean_pipeline_score_results(self) -> pd.DataFrame:
        """Compute mean score results for each pipeline combination and hyperparameter combination.

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with mean score results for each pipeline combination and each parameter combination,
            sorted by the highest mean score.

        Notes
        -----
        The pipeline with the highest "mean over the mean test scores" does not necessarily correspond to the
        best-performing pipeline as returned by
        :func:`~biopsykit.classification.model_selection.SklearnPipelinePermuter.metric_summary` or
        :func:`~biopsykit.classification.model_selection.SklearnPipelinePermuter.best_estimator_summary` because the
        best-performing pipelines are determined by averaging the `best_estimator` instances, as determined by
        `scikit-learn` over all folds. Hence, all `best_estimator` instances can have a **different** set of
        hyperparameters.

        This function should only be used if you want to gain a deeper understanding of the different hyperparameter
        combinations and their performance. If you want to get the best-performing pipeline(s) to report in a paper,
        use :func:`~biopsykit.classification.model_selection.SklearnPipelinePermuter.metric_summary` or
        :func:`~biopsykit.classification.model_selection.SklearnPipelinePermuter.best_estimator_summary` instead.

        """
        score_results = self.pipeline_score_results()
        score_summary_mean = (
            score_results.groupby(score_results.index.names[:-1])
            .agg(["mean", "std"])
            .sort_values(by=(f"mean_test_{self.refit}", "mean"), ascending=False)
        )
        return score_summary_mean

    def best_hyperparameter_pipeline(self) -> pd.DataFrame:
        """Return the evaluation results for the pipeline with the best-performing hyperparameter set.

        This returns the pipeline with the **unique** hyperparameter combination that achieved
        the highest mean score over all outer folds.

        Notes
        -----
        This `best pipeline` does not necessarily correspond to the overall best-performing pipeline as returned by
        :func:`~biopsykit.classification.model_selection.SklearnPipelinePermuter.metric_summary` or
        :func:`~biopsykit.classification.model_selection.SklearnPipelinePermuter.best_estimator_summary` because the
        best-performing pipelines are determined by averaging the `best_estimator` instances, as determined by
        `scikit-learn` over all folds. Hence, all `best_estimator` instances can have a **different** set of
        hyperparameters. This function returns the pipeline with the **unique** hyperparameter combination that
        achieved the highest mean score over all outer folds.

        This function should only be used if you want to gain a deeper understanding of the different hyperparameter
        combinations and their performance. If you want to get the best-performing pipeline(s) to report in a paper,
        use :func:`~biopsykit.classification.model_selection.SklearnPipelinePermuter.metric_summary` or
        :func:`~biopsykit.classification.model_selection.SklearnPipelinePermuter.best_estimator_summary` instead.


        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with the evaluation results of the best pipeline over all outer folds

        """
        score_summary = self.pipeline_score_results()
        score_summary_mean = self.mean_pipeline_score_results()
        return score_summary.loc[score_summary_mean.index[0]].dropna(how="all", axis=1)

    @staticmethod
    def _check_missing_params(
        model_dict: dict[str, dict[str, BaseEstimator]],
        param_dict: dict[str, Optional[Union[Sequence[dict[str, Any]], dict[str, Any]]]],
    ):
        for _category, estimator_dict in model_dict.items():
            if not set(estimator_dict.keys()).issubset(set(param_dict.keys())):
                missing_params = list(set(estimator_dict.keys()) - set(param_dict.keys()))
                raise ValueError(f"Some estimators are missing parameters: {missing_params}")

    def metric_summary_to_latex(
        self,
        data: Optional[pd.DataFrame] = None,
        metrics: Optional[Sequence[str]] = None,
        pipeline_steps: Optional[Sequence[str]] = None,
        si_table_format: Optional[str] = None,
        highlight_best: Optional[Union[str, bool]] = None,
        **kwargs,
    ) -> str:
        """Return a latex table with the performance metrics of the pipeline combinations.

        By default, this function uses the attribute of the ``SklearnPipelinePermuter`` instance.
        If the ``data`` parameter is set, the function uses the dataframe passed as argument.

        Parameters
        ----------
        data : :class:`~pandas.DataFrame`, optional
            dataframe with performance metrics if custom data should be used or ``None`` to use the attribute of the
            ``SklearnPipelinePermuter`` instance. Default: ``None``
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
        kwargs.setdefault("clines", "skip-last;data")
        kwargs.setdefault("hrules", True)
        kwargs.setdefault("position", "ht!")
        kwargs.setdefault("position_float", "centering")
        kwargs.setdefault("siunitx", True)
        if si_table_format is None:
            si_table_format = "table-format = 2.1(2)"

        if data is None:
            data = self.metric_summary()
        metric_summary = data.copy()

        if pipeline_steps is None:
            if isinstance(metric_summary.index, pd.MultiIndex):
                pipeline_steps = list(metric_summary.index.names)
            else:
                pipeline_steps = [metric_summary.index.name]

        if metrics is None:
            metrics = metric_summary.filter(like="mean_test").columns
            # extract metric names
            metrics = [m.split("_")[-1] for m in metrics]

        levels_to_drop = [step for step in metric_summary.index.names if step not in pipeline_steps]
        metric_summary = metric_summary.droplevel(levels_to_drop)
        metric_summary = metric_summary.rename(index=clf_map)

        list_metric_summary = []
        for metric in metrics:
            list_metric_summary.append(metric_summary.filter(regex=f"(mean|std)_test_{metric}"))

        metric_summary = pd.concat(list_metric_summary, axis=1)

        # convert to percent
        metric_summary = metric_summary * 100
        metric_summary_export = metric_summary.copy()

        for metric in metrics:
            mean_test = f"mean_test_{metric}"
            std_test = f"std_test_{metric}"
            m_sd = metric_summary_export.apply(
                lambda x, m_t=mean_test, std_t=std_test: rf"{x[m_t]:.1f}({x[std_t]:.1f})", axis=1
            )
            metric_summary_export = metric_summary_export.assign(**{metric: m_sd})
        metric_summary_export = metric_summary_export[metrics].copy()

        if isinstance(metric_summary_export.index, pd.MultiIndex):
            metric_summary_export.index = metric_summary_export.index.rename(pipeline_step_map)
        metric_summary_export = metric_summary_export.rename(columns=metric_map)

        kwargs.setdefault("column_format", self._format_latex_column_format(metric_summary_export))

        styler = metric_summary_export.style
        styler = self._highlight_best(metric_summary, styler, highlight_best, metric_summary_export)

        metric_summary_tex = styler.to_latex(**kwargs)
        metric_summary_tex = self._apply_latex_code_correction(metric_summary_tex, si_table_format)
        return metric_summary_tex

    @staticmethod
    def _format_latex_column_format(data: pd.DataFrame):
        column_format = "l" * data.index.nlevels
        if isinstance(data.columns, pd.MultiIndex):
            ncols = len(data.columns)
            ncols_last_level = len(data.columns.get_level_values(-1).unique())
            column_format += ("S" * ncols_last_level + "|") * (ncols // ncols_last_level)
            # remove the last "|"
            column_format = column_format[:-1]
        else:
            column_format += "S" * len(data.columns)
        return column_format

    @staticmethod
    def _apply_latex_code_correction(table: str, si_table_format: str) -> str:
        if si_table_format is not None:
            table = re.sub(r"(\\begin\{tabular\})", r"\\sisetup{" + si_table_format + r"}\n\n\1", table)
        return table

    def to_pickle(self, file_path: path_t) -> None:
        """Export the current instance as a pickle file.

        Parameters
        ----------
        file_path : :class:`~pathlib.Path` or str
            file path to export

        """
        file_path = Path(file_path)
        _assert_file_extension(file_path, ".pkl")
        with file_path.open(mode="wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(file_path: path_t) -> "SklearnPipelinePermuter":
        """Import a ``SklearnPipelinePermuter`` instance from a pickle file.

        Parameters
        ----------
        file_path : :class:`~pathlib.Path` or str
            file path to import

        Returns
        -------
        :class:`~biopsykit.classification.model_selection.SklearnPipelinePermuter`
            ``SklearnPipelinePermuter` instance

        """
        file_path = Path(file_path)
        _assert_file_extension(file_path, ".pkl")
        with file_path.open(mode="rb") as f:
            return pickle.load(f)

    def update_permuter(
        self,
        model_dict: Optional[dict[str, dict[str, BaseEstimator]]] = None,
        param_dict: Optional[dict[str, Any]] = None,
        hyper_search_dict: Optional[dict[str, dict[str, Any]]] = None,
    ) -> Self:
        """Update the ``SklearnPipelinePermuter`` instance with new model and parameter dictionaries.

        Parameters
        ----------
        model_dict : dict, optional
            dictionary with model classes for each pipeline step
        param_dict : dict, optional
            dictionary with parameter grids for each pipeline step
        hyper_search_dict : dict, optional
            dictionary with hyperparameter search settings for each estimator

        Returns
        -------
        :class:`~biopsykit.classification.model_selection.SklearnPipelinePermuter`
            updated ``SklearnPipelinePermuter`` instance

        """
        permuter = SklearnPipelinePermuter(model_dict, param_dict, hyper_search_dict)
        return SklearnPipelinePermuter._merge_permuter_params(self, permuter)

    @classmethod
    def _merge_permuter_params(cls, permuter_01: Self, permuter_02: Self):
        # merge model dicts
        permuter_01.models = merge_nested_dicts(permuter_01.models, permuter_02.models)
        # merge hyperparameter search dicts
        permuter_01.hyper_search_dict = merge_nested_dicts(permuter_01.hyper_search_dict, permuter_02.hyper_search_dict)

        # merge hyperparameter dicts
        permuter_01.param_searches = merge_nested_dicts(permuter_01.param_searches, permuter_02.param_searches)
        permuter_01.params = merge_nested_dicts(permuter_01.params, permuter_02.params)

        # merge model combinations
        permuter_01.model_combinations += permuter_02.model_combinations
        permuter_01.model_combinations = list(set(permuter_01.model_combinations))

        return permuter_01

    @classmethod
    def merge_permuter_instances(cls, permuter: Union[Sequence[Self], Sequence[path_t]]) -> Self:
        """Merge two (or more) :class:`~biopsykit.classification.model_selection.SklearnPipelinePermuter` instances.

        This function expects at least two ``SklearnPipelinePermuter`` instances to merge. The function first performs
        a deep copy of the first instance and then merges all attributes of the remaining ``permuter`` instance with
        the copy. The ``permuter`` instances passed to this function are not modified.

        Parameters
        ----------
        permuter : list of :class:`~biopsykit.classification.model_selection.SklearnPipelinePermuter` instances or
                    list of file paths to pickled `SklearnPipelinePermuter` instances

        Returns
        -------
        :class:`~biopsykit.classification.model_selection.SklearnPipelinePermuter`
            merged ``SklearnPipelinePermuter`` instance

        """
        # ensure that permuter contains at least two instances
        if len(permuter) < 2:
            raise ValueError("At least two SklearnPipelinePermuter instances must be passed to this function.")

        if all(isinstance(p, (str, Path)) for p in permuter):
            permuter = [cls.from_pickle(p) for p in permuter]

        # make deep copy of first instance
        base_permuter = deepcopy(permuter[0])

        for p in permuter[1:]:
            if base_permuter.scoring != p.scoring:
                raise ValueError(
                    f"Cannot merge permuter instances with different scoring functions: "
                    f"{base_permuter.scoring} vs. {p.scoring}"
                )
            if base_permuter.refit != p.refit:
                raise ValueError(
                    f"Cannot merge permuter instances with different refit options: {base_permuter.refit} vs. {p.refit}"
                )

            SklearnPipelinePermuter._merge_permuter_params(base_permuter, p)

            # merge results dataframes
            results_concat = pd.concat([base_permuter.results, p.results], axis=0)
            param_cols = list(results_concat.filter(like="param_").columns)
            # drop duplicate parameter combinations in results
            results_concat = results_concat.reset_index("outer_fold").drop_duplicates(
                subset=["outer_fold", *param_cols]
            )
            results_concat = results_concat.set_index("outer_fold", append=True)
            base_permuter.results = results_concat

        return base_permuter

    @staticmethod
    def _apply_score(row: pd.Series, score_func, pos_label: str):
        true_labels_folds = row[0]
        predicted_labels_folds = row[1]

        kwargs = {}
        params = signature(score_func).parameters
        if "pos_label" in params:
            kwargs["pos_label"] = pos_label
        if "zero_division" in params:
            kwargs["zero_division"] = 0

        scores = [
            score_func(true_labels, predicted_labels, **kwargs)
            for true_labels, predicted_labels in zip(true_labels_folds, predicted_labels_folds)
        ]
        return pd.Series(scores)

    def compute_additional_metrics(self, metric_summary: pd.DataFrame, metrics: str_t, pos_label: str) -> pd.DataFrame:
        """Compute additional classification metrics.

        Parameters
        ----------
        metric_summary : :class:`~pandas.DataFrame`
            metric summary from :meth:`~biopsykit.classification.model_selection.SklearnPipelinePermuter.metric_summary`
        metrics : str or list of str
            metric(s) to compute
        pos_label : str
            positive label for binary classification

        Returns
        -------
        :class:`~pandas.DataFrame`
            metric summary with additional metrics computed

        """
        if isinstance(metrics, str):
            metrics = [metrics]
        metric_slice = metric_summary[["true_labels_folds", "predicted_labels_folds"]].copy()
        metric_out = {}
        score_funcs = dict(getmembers(sklearn.metrics))
        for metric in metrics:
            if metric.endswith("_score"):
                score_name = metric
                # strip '_score' suffix from metric name for column name
                metric = metric.replace("_score", "")  # noqa: PLW2901
            else:
                # name for calling sklearn metric function
                score_name = metric + "_score"

            if score_name in score_funcs:
                score_func = score_funcs[score_name]
            else:
                raise ValueError(f"Metric '{metric}' not found.")

            metric_out[metric] = metric_slice.apply(self._apply_score, args=(score_func, pos_label), axis=1)
        metric_out = pd.concat(metric_out, names=["score", "folds"], axis=1)

        metric_out = metric_out.stack(["score", "folds"])
        metric_out = metric_out.groupby(metric_out.index.names[:-1]).agg(
            [("mean", lambda x: np.mean(x)), ("std", lambda x: np.std(x))]
        )

        metric_out = metric_out.unstack("score").sort_index(axis=1, level="score")
        metric_out.columns = metric_out.columns.map("_test_".join)
        metric_summary = metric_summary.join(metric_out)

        # resort columns so that all "mean_test_*" and "std_test_*" columns are at the end
        cols = list(metric_summary.filter(regex="^(?!mean_test_|std_test_).*$").columns)
        cols += list(metric_summary.filter(regex="^(mean_test_|std_test_).*$").columns)
        metric_summary = metric_summary[cols]

        return metric_summary

    @staticmethod
    def _highlight_best(metric_summary, styler, highlight_best, metric_summary_export):
        if isinstance(highlight_best, str):
            max_metric = metric_summary[f"mean_test_{highlight_best}"].idxmax()
            # get index of max metric
            max_metric = metric_summary_export.index.get_loc(max_metric)
            styler = styler.highlight_max(subset=metric_map[highlight_best], props="bfseries: ;")
            # get maximum of metric_summary
            # make index bold
            styler = styler.apply_index(lambda x: np.where(x.index == max_metric, "bfseries: ;", ""))
        elif isinstance(highlight_best, bool) and highlight_best:
            styler = styler.highlight_max(props="bfseries: ;")
        return styler
