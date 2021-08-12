"""Module for systematically evaluating different combinations of sklearn pipelines."""
from itertools import product
from pathlib import Path
from shutil import rmtree
from typing import Optional, Dict, Any, Sequence, Union, Tuple

from joblib import Memory

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from biopsykit.classification.utils import _PipelineWrapper
from biopsykit.classification.model_selection import nested_cv_grid_search
from biopsykit.utils._datatype_validation_helper import _assert_file_extension
from biopsykit.utils._types import path_t, T

__all__ = ["SklearnPipelinePermuter"]


class SklearnPipelinePermuter:
    """Class for systematically evaluating different sklearn pipeline combinations."""

    def __init__(
        self,
        model_dict: Optional[Dict[str, Dict[str, BaseEstimator]]] = None,
        param_dict: Optional[Dict[str, Optional[Union[Sequence[Dict[str, Any]], Dict[str, Any]]]]] = None,
        **kwargs,
    ):
        """Class for systematically evaluating different sklearn pipeline combinations.

        This class can be used to, for instance, evaluate combinations of different feature selection methods
        (e.g., :class:`~sklearn.feature_selection.SelectKBest`,
        :class:`~sklearn.feature_selection.SequentialFeatureSelector`) with different estimators
        (e.g., :class:`~sklearn.svm.SVC`, :class:`~sklearn.tree.DecisionTreeClassifier`), any much more.

        For all combinations, hyperparameter optimization can be performed in a grid-search by passing one joint
        parameter grid (see Examples).

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
        >>> pipeline_permuter = SklearnPipelinePermuter(model_dict, param_dict)
        >>> pipeline_permuter.fit(X, y, outer_cv=KFold(), inner_cv=KFold())

        """
        self.models: Dict[str, Dict[str, BaseEstimator]] = {}
        """Dictionary with pipeline steps and the different transformers/estimators per step."""

        self.params: Dict[str, Optional[Union[Sequence[Dict[str, Any]], Dict[str, Any]]]] = {}
        """Dictionary with parameter sets to test for the different transformers/estimators per pipeline step."""

        self.model_combinations: Sequence[Tuple[Tuple[str, str], ...]] = []
        """List of model combinations, i.e. permutations of the different transformers/estimators for
        each pipeline step."""

        self.grid_searches: Dict[Tuple[str, str], pd.DataFrame] = {}
        """Dictionary with grid search search results for each pipeline step combination."""

        self.results: Optional[pd.DataFrame] = None
        """Dataframe with parameter search results of each pipeline step combination."""

        self.scoring: str = ""

        if kwargs.get("score_summary") is not None:
            self.results = kwargs.get("score_summary")
        else:
            for category in model_dict:
                if not set(model_dict[category].keys()).issubset(set(param_dict.keys())):
                    missing_params = list(set(model_dict[category].keys()) - set(param_dict.keys()))
                    raise ValueError("Some estimators are missing parameters: {}".format(missing_params))

            model_combinations = list(
                product(*[[(step, k) for k in list(model_dict[step].keys())] for step in model_dict])
            )

            # assert that all entries of the param dict are lists for uniform handling
            for k, v in param_dict.items():
                if isinstance(v, dict):
                    param_dict[k] = [v]

            self.models = model_dict
            self.params = param_dict
            self.model_combinations = model_combinations

    @classmethod
    def from_csv(cls: T, file_path: path_t, num_pipeline_steps: Optional[int] = 3) -> T:
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
        return cls(score_summary=score_summary)

    def fit(  # pylint:disable=invalid-name
        self,
        X: np.ndarray,  # noqa
        y: np.ndarray,
        outer_cv: BaseCrossValidator,
        inner_cv: BaseCrossValidator,
        scoring: Optional[str] = None,
        **kwargs,
    ):
        """Run fit for all pipeline combinations and sets of parameters.

        This function calls :func:`~biopsykit.classification.model_selection.nested_cv_grid_search` for all
        Pipeline combinations and stores the results in the ``grid_searches`` attribute.

        Parameters
        ----------
        X : array-like of shape (`n_samples`, `n_features`)
            Training vector, where `n_samples` is the number of samples and `n_features` is the number of features.
        y : array-like of shape (`n_samples`, `n_output`) or (`n_samples`,)
            Target (i.e., class labels) relative to X for classification or regression.
        outer_cv : `CV splitter`_
            Cross-validation object determining the cross-validation splitting strategy of the outer cross-validation.
        inner_cv : `CV splitter`_
            Cross-validation object determining the cross-validation splitting strategy of the grid-search.
        scoring : str, optional
            A str specifying the scoring metric to use for evaluation.
        **kwargs :
            additional arguments that are passed to
            :func:`~biopsykit.classification.model_selection.nested_cv_grid_search` and
            :class:`~sklearn.model_selection.GridSearchCV`

        """
        self.results = None
        if scoring is None:
            scoring = "accuracy"
        self.scoring = scoring
        kwargs.setdefault("n_jobs", -1)
        kwargs.setdefault("verbose", 1)
        kwargs.setdefault("error_score", "raise")

        # Create a temporary folder to store the transformers of the pipeline
        location = "cachedir"
        memory = Memory(location=location, verbose=0)

        for model_combination in tqdm(self.model_combinations):
            if model_combination in self.grid_searches:
                # continue if we already tried this combination
                continue

            pipeline_params = [(m, self.params[k[1]]) for m, k in zip(self.models.keys(), model_combination)]
            pipeline_params = list(filter(lambda p: p[1] is not None, pipeline_params))
            pipeline_params = [(m, k_new) for m, k in pipeline_params for k_new in k if k is not None]

            cats = {p[0] for p in pipeline_params}
            pipeline_params = [list(filter(lambda p, c=cat: p[0] == c, pipeline_params)) for cat in cats]
            pipeline_params = list(product(*pipeline_params))

            pipeline_params = [
                tuple({"{}__{}".format(step[0], k): v for k, v in step[1].items()} for step in combi)
                for combi in pipeline_params
            ]
            pipeline_params = [{k: v for x in param for k, v in x.items()} for param in pipeline_params]

            print(
                "### Running GridSearchCV for pipeline: {} with {} parameter grid(s):".format(
                    model_combination, len(pipeline_params)
                )
            )

            for i, param_dict in enumerate(pipeline_params):
                print("Parameter grid #{}: {}".format(i, param_dict))
                model_cls = [(step, self.models[step][m]) for step, m in model_combination]
                pipeline = Pipeline(model_cls, memory=memory)

                result_dict = nested_cv_grid_search(
                    X,
                    y,
                    param_dict=param_dict,
                    pipeline=pipeline,
                    outer_cv=outer_cv,
                    inner_cv=inner_cv,
                    scoring=scoring,
                    **kwargs,
                )

                self.grid_searches[model_combination] = result_dict
                print("")
            print("")

        # Delete the temporary cache before exiting
        memory.clear(warn=False)
        rmtree(location)

    def pipeline_score_results(self) -> pd.DataFrame:
        """Return parameter search results for each pipeline combination.

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with parameter search results for each pipeline combination

        """
        if self.results is not None:
            return self.results

        gs_param_list = []
        for param, gs in self.grid_searches.items():
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
            param_dict = {"pipeline_{}".format(key): val for key, val in param}
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

    def mean_pipeline_score_results(self) -> pd.DataFrame:
        """Compute mean score results for each pipeline combination.

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with mean score results for each pipeline combination and each parameter combination,
            sorted by the highest mean score.

        """
        score_results = self.pipeline_score_results()
        score_summary_mean = (
            score_results.groupby(score_results.index.names[:-1])
            .agg(["mean", "std"])
            .sort_values(by=("mean_test_{}".format(self.scoring), "mean"), ascending=False)
        )
        return score_summary_mean

    def best_pipeline(self) -> pd.DataFrame:
        """Return the evaluation results for the `overall best pipeline`.

        The `overall best pipeline` is the pipeline with the parameter combination that achieved the highest mean
        score over all outer folds.

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with the evaluation results of the best pipeline over all outer folds

        """
        score_summary = self.pipeline_score_results()
        score_summary_mean = self.mean_pipeline_score_results()
        return score_summary.loc[score_summary_mean.index[0]].dropna(how="all", axis=1)

    def metric_summary(self) -> pd.DataFrame:
        """Return a summary with all performance metrics for the `best estimator` of each pipeline combination.

        The `best estimator` for each pipeline combination is the best estimator that
        :class:`~sklearn.model_selection.GridSearchCV` returns for each outer fold, i.e. the pipeline which yielded
        the highest average test score (over all inner folds).

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with performance metric summary the `best estimator` of each pipeline combination.

        """
        list_metric_summary = []
        for param_key in self.grid_searches:
            param_dict = {"pipeline_{}".format(key): val for key, val in param_key}
            conf_matrix = np.sum(self.grid_searches[param_key]["conf_matrix"], axis=0)
            true_labels = np.array(self.grid_searches[param_key]["true_labels"]).ravel()
            predicted_labels = np.array(self.grid_searches[param_key]["predicted_labels"]).ravel()
            df_metric = pd.DataFrame(param_dict, index=[0])
            df_metric["conf_matrix"] = [list(conf_matrix.flatten())]
            df_metric["true_labels"] = [true_labels]
            df_metric["predicted_labels"] = [predicted_labels]

            for key in self.grid_searches[param_key]:
                if "test" in key:
                    test_scores = self.grid_searches[param_key][key]
                    df_metric["mean_{}".format(key)] = np.mean(test_scores)
                    df_metric["std_{}".format(key)] = np.std(test_scores)
                    df_metric[["{}_fold_{}".format(key, i) for i in range(len(test_scores))]] = list(test_scores)

            df_metric = df_metric.set_index(list(df_metric.columns)[: len(param_dict)])
            list_metric_summary.append(df_metric)

        return pd.concat(list_metric_summary)

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
        be_list = []
        for param_key in self.grid_searches:
            param_dict = {"pipeline_{}".format(key): val for key, val in param_key}
            df_be = pd.DataFrame(param_dict, index=[0])
            df_be["best_estimator"] = _PipelineWrapper(self.grid_searches[param_key]["best_estimator"])
            df_be = df_be.set_index(list(df_be.columns)[:-1])
            be_list.append(df_be)

        return pd.concat(be_list)
