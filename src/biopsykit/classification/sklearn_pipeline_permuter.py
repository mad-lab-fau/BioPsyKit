from itertools import product
from shutil import rmtree
from typing import Optional, Dict, Any, Sequence, Union
from joblib import Memory

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from tqdm.notebook import tqdm

from biopsykit.classification.utils import _PipelineWrapper
from biopsykit.classification.nested_cv import nested_cv_grid_search
from biopsykit.utils._datatype_validation_helper import _assert_file_extension
from biopsykit.utils._types import path_t, T

__all__ = ["SklearnPipelinePermuter"]


class SklearnPipelinePermuter:
    def __init__(
        self,
        models: Optional[Dict[str, Dict[str, BaseEstimator]]] = None,
        params: Optional[Dict[str, Optional[Union[Sequence[Dict[str, Any]], Dict[str, Any]]]]] = None,
        score_summary: Optional[pd.DataFrame] = None,
    ):
        if score_summary is not None:
            self.df_score_summary = score_summary
        else:
            for category in models:
                if not set(models[category].keys()).issubset(set(params.keys())):
                    missing_params = list(set(models[category].keys()) - set(params.keys()))
                    raise ValueError("Some estimators are missing parameters: {}".format(missing_params))

            model_combinations = list(product(*[[(step, k) for k in list(models[step].keys())] for step in models]))

            # assert that all entries of the param dict are lists for uniform handling
            for k, v in params.items():
                if isinstance(v, dict):
                    params[k] = [v]

            self.models = models
            self.params = params
            self.model_combinations = model_combinations
            self.grid_searches = {}
            self.df_score_summary: Optional[pd.DataFrame] = None

    @classmethod
    def from_csv(cls: T, file_path: path_t, num_steps: Optional[int] = 3) -> T:
        score_summary = pd.read_csv(file_path)
        score_summary = score_summary.set_index(list(score_summary.columns)[: num_steps + 2])
        return cls(score_summary=score_summary)

    def fit(self, X, y, outer_cv: BaseCrossValidator, inner_cv: BaseCrossValidator, **kwargs):
        self.df_score_summary = None
        kwargs.setdefault("n_jobs", -1)
        kwargs.setdefault("verbose", 1)
        kwargs.setdefault("scoring", None)
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

            cats = set([p[0] for p in pipeline_params])
            pipeline_params = [list(filter(lambda p: p[0] == cat, pipeline_params)) for cat in cats]
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
                    X, y, params=param_dict, pipeline=pipeline, outer_cv=outer_cv, inner_cv=inner_cv, **kwargs
                )

                self.grid_searches[model_combination] = result_dict
                print("")
            print("")

        # Delete the temporary cache before exiting
        memory.clear(warn=False)
        rmtree(location)

    def score_summary(self):
        if self.df_score_summary is not None:
            return self.df_score_summary

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
        self.df_score_summary = df_summary.sort_index().sort_index(axis=1)
        return self.df_score_summary

    def export_score_summary(self, file_path: path_t):
        _assert_file_extension(file_path, ".csv")
        self.score_summary().to_csv(file_path)

    def score_summary_mean(self):
        score_summary = self.score_summary()
        score_summary_mean = (
            score_summary.groupby(score_summary.index.names[:-1])
            .agg(["mean", "std"])
            .sort_values(by=("mean_test_score", "mean"), ascending=False)
        )
        return score_summary_mean

    def best_pipeline(self):
        score_summary = self.score_summary()
        score_summary_mean = self.score_summary_mean()
        return score_summary.loc[score_summary_mean.index[0]].dropna(how="all", axis=1)

    def metric_summary(self):
        conf_matrix_list = []
        for param_key in self.grid_searches:
            param_dict = {"pipeline_{}".format(key): val for key, val in param_key}
            conf_matrix = np.sum(self.grid_searches[param_key]["conf_matrix"], axis=0)
            test_scores = self.grid_searches[param_key]["test_score"]
            df_metric = pd.DataFrame(param_dict, index=[0])
            df_metric["conf_matrix"] = [list(conf_matrix.flatten())]
            df_metric["mean_test_score"] = np.mean(test_scores)
            df_metric["std_test_score"] = np.std(test_scores)
            df_metric[["fold_{}".format(i) for i in range(len(test_scores))]] = list(test_scores)

            df_metric = df_metric.set_index(list(df_metric.columns)[: len(param_dict)])
            conf_matrix_list.append(df_metric)

        return pd.concat(conf_matrix_list)

    def export_metric_summary(self, file_path: path_t):
        _assert_file_extension(file_path, ".csv")
        self.metric_summary().to_csv(file_path, sep=";")

    def best_estimator_summary(self):
        be_list = []
        for param_key in self.grid_searches:
            param_dict = {"pipeline_{}".format(key): val for key, val in param_key}
            df_be = pd.DataFrame(param_dict, index=[0])
            df_be["best_estimator"] = _PipelineWrapper(self.grid_searches[param_key]["best_estimator"])
            df_be = df_be.set_index(list(df_be.columns)[:-1])

            be_list.append(df_be)

        return pd.concat(be_list)
