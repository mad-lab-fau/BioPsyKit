from typing import Dict, Sequence, Union, Tuple, Literal, Optional

import pandas as pd
import pingouin as pg
from IPython.core.display import display, Markdown
from biopsykit._types import path_t

MAP_STAT_TESTS = {
    "normality": pg.normality,
    "equal_var": pg.homoscedasticity,
    "anova": pg.anova,
    "welch_anova": pg.welch_anova,
    "rm_anova": pg.rm_anova,
    "mixed_anova": pg.mixed_anova,
    "kruskal": pg.kruskal,
    "pairwise_ttests": pg.pairwise_ttests,
    "pairwise_tukey": pg.pairwise_tukey,
    "pairwise_gameshowell": pg.pairwise_gameshowell,
}

MAP_STAT_PARAMS = {
    "normality": ["dv", "group"],
    "equal_var": ["dv", "group"],
    "anova": ["dv", "between"],
    "welch_anova": ["dv", "between"],
    "rm_anova": ["dv", "within", "subject"],
    "mixed_anova": ["dv", "between", "within", "subject"],
    "kruskal": ["dv", "between"],
    "pairwise_ttests": [
        "dv",
        "between",
        "within",
        "subject",
        "effsize",
        "tail",
        "padjust",
    ],
    "pairwise_tukey": ["dv", "between", "effsize"],
    "pairwise_gameshowell": ["dv", "between", "effsize"],
}

MAP_NAMES = {
    "normality": "Test for Normal Distribution",
    "equal_var": "Test for Homoscedasticity (Equal Variances)",
    "anova": "ANOVA",
    "welch_anova": "Welch ANOVA",
    "rm_anova": "Repeated-measurement ANOVA",
    "mixed_anova": "Mixed ANOVA",
    "kruskal": "Kruskal-Wallis H-test for independent samples",
    "pairwise_ttests": "Pairwise t-Tests",
    "pairwise_tukey": "Pairwise Tukey's HSD (Honestly Significant Differences) Test",
    "pairwise_gameshowell": "Pairwise Games-Howell post-hoc Test",
}

MAP_CATEGORIES = {
    "prep": "Preparatory Analysis",
    "test": "Statistical Tests",
    "posthoc": "Post-Hoc Analysis",
}

MAP_LATEX_EXPORT = {
    "anova": ["ddof1", "ddof2", "F", "p-unc", "np2"],
    "welch_anova": ["ddof1", "ddof2", "F", "p-unc", "np2"],
}

MAP_LATEX = {
    "ddof1": r"$\text{df}_{Num}$",
    "ddof2": r"$\text{df}_{Den}$",
    "F": "F",
    "p-unc": "p",
    "np2": r"$\eta^2_p$",
}

STATS_CATEGORY = Literal["test", "posthoc"]
STATS_TYPE = Literal["between", "within", "interaction"]
PLOT_TYPE = Literal["single", "multi"]

_sig_cols = ["p-corr", "p-tukey", "p-unc", "pval"]


class StatsPipeline:
    def __init__(self, steps: Sequence[Tuple[str, str]], params: Dict[str, str]):
        self.steps = steps
        self.params = params
        self.results: Union[None, Dict[str, pd.DataFrame]] = None
        self.category_steps = {}
        for step in self.steps:
            self.category_steps.setdefault(step[0], [])
            self.category_steps[step[0]].append(step[1])

    def results_cat(self, category: str):
        cat = self.category_steps.get(category, [])
        if len(cat) == 1:
            return self.results[cat[0]]
        elif len(cat) > 1:
            return {c: self.results[c] for c in cat}
        return {}

    def apply(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        pipeline_results = {}
        data = data.reset_index()

        for step in self.steps:

            general_params = {
                key: value
                for key, value in self.params.items()
                if len(key.split("__")) == 1
            }
            specific_params = {
                key.split("__")[1]: value
                for key, value in self.params.items()
                if len(key.split("__")) > 1 and step[0] in key.split("__")
            }
            params = {
                key: general_params[key]
                for key in MAP_STAT_PARAMS[step[1]]
                if key in general_params
            }

            grouper = None
            if "groupby" in specific_params:
                grouper = specific_params.pop("groupby")
            elif "groupby" in general_params:
                grouper = general_params.pop("groupby")

            if grouper:
                result = data.groupby(grouper).apply(
                    lambda df: MAP_STAT_TESTS[step[1]](
                        data=df, **specific_params, **params
                    )
                )
            else:
                result = MAP_STAT_TESTS[step[1]](data=data, **params)

            if (
                step[0] == "posthoc"
                and "padjust" in general_params
                and "padjust" not in params
            ):
                # apply p-adjustment for posthoc testing if it was specified in the pipeline
                # but do it only manually if it's not supported by the test function
                # (otherwise it would be in the 'params' dict)
                result = self.multicomp(result, method=general_params["padjust"])

            pipeline_results[step[1]] = result

        self.results = pipeline_results
        return pipeline_results

    def _ipython_display_(self):
        display(self._param_df().T)
        display(self._result_df().T)

    def display_results(self, **kwargs):
        sig_only = kwargs.get("sig_only", {})
        if sig_only is None:
            sig_only = {}
        if isinstance(sig_only, str):
            if sig_only == "all":
                sig_only = True
            else:
                sig_only = [sig_only]
        if isinstance(sig_only, bool):
            sig_only = {cat: sig_only for cat in self.category_steps.keys()}
        if isinstance(sig_only, list):
            sig_only = {cat: cat in sig_only for cat in self.category_steps.keys()}

        if self.results is None:
            display(Markdown("No results."))
            return

        display(Markdown("""<font size="4"><b> Overview </b></font>"""))
        display(self)
        for category, steps in self.category_steps.items():
            if kwargs.get(category, True):
                display(
                    Markdown(
                        """<font size="4"><b> {} </b></font>""".format(
                            MAP_CATEGORIES[category]
                        )
                    )
                )
                for step in steps:
                    display(Markdown("**{}**".format(MAP_NAMES[step])))
                    df = self.results[step]
                    if sig_only.get(category, False):
                        df = self._filter_sig(df)
                        if df.empty:
                            display(Markdown("*No significant p-values.*"))
                            continue
                    display(df)

    def _filter_sig(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in _sig_cols:
            if col in data.columns:
                if data[col].isna().all():
                    # drop column if all values are NaN => most probably because we turned on p-adjust but only
                    # have two main effects
                    data = data.drop(columns=col)
                    continue
                return data[data[col] < 0.05]

    def _filter_pcol(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        for col in _sig_cols:
            if col in data.columns:
                return data[col]
        return None

    def _filter_effect(
        self, stats_category: STATS_CATEGORY, stats_type: STATS_TYPE
    ) -> pd.DataFrame:
        results = self.results_cat(stats_category)
        if "Contrast" in results.columns:
            if stats_type == "interaction":
                key = "{} * {}".format(self.params["within"], self.params["between"])
            else:
                key = self.params[stats_type]

            results = results[results["Contrast"] == key]
            results = results.drop(columns="Contrast")

        return results

    def export_statistics(self, file_path: path_t):
        writer = pd.ExcelWriter(file_path, engine="xlsxwriter")
        workbook = writer.book
        header_format = workbook.add_format({"bold": True})
        param_df = self._param_df()
        param_df.to_excel(writer, sheet_name="parameter")
        for key, df in self.results.items():
            df.to_excel(writer, sheet_name=key, startrow=1)
            worksheet = writer.sheets[key]
            worksheet.write_string(0, 0, MAP_NAMES[key], header_format)
        writer.save()

    def sig_brackets(
        self,
        stats_category: STATS_CATEGORY,
        stats_type: STATS_TYPE,
        plot_type: Optional[PLOT_TYPE] = "single",
        features: Optional[
            Union[str, Sequence[str], Dict[str, Union[str, Sequence[str]]]]
        ] = None,
        x: Optional[str] = None,
        subplots: Optional[bool] = False,
    ):
        if isinstance(features, str):
            features = [features]

        stats_data = self._filter_effect(stats_category, stats_type)
        stats_data = self._filter_sig(stats_data)

        if stats_type == "interaction":
            stats_data = stats_data.set_index(self.params["within"])
            box_pairs = stats_data.apply(
                lambda row: ((row.name, row["A"]), (row.name, row["B"])), axis=1
            )
        else:
            if self.params[stats_type] in stats_data.columns:
                stats_data = stats_data.reset_index().set_index(self.params[stats_type])
            if features is not None:
                stats_data = pd.concat(
                    [stats_data.filter(like=f, axis=0) for f in features]
                )
            if plot_type == "single":
                box_pairs = stats_data.apply(lambda row: (row["A"], row["B"]), axis=1)
            else:
                if x is None:
                    raise ValueError(
                        "`x` must be specified when `plot_type` is `multi`!"
                    )
                stats_data = stats_data.reset_index().set_index(x)
                box_pairs = stats_data.apply(
                    lambda row: ((row.name, row["A"]), (row.name, row["B"])), axis=1
                )

        if box_pairs.empty:
            return [], []

        box_pairs = list(box_pairs)
        pvalues = list(self._filter_pcol(stats_data))

        if subplots:
            dict_box_pairs = {}
            dict_pvalues = {}
            if isinstance(features, list):
                features = {f: f for f in features}
            for key in features:
                features_list = features[key]
                if isinstance(features_list, str):
                    features_list = [features_list]

                list_pairs = dict_box_pairs.setdefault(key, [])
                list_pvalues = dict_pvalues.setdefault(key, [])
                for i, sig_pair in enumerate(box_pairs):
                    if sig_pair[0][0] in features_list:
                        list_pairs.append(sig_pair)
                        list_pvalues.append(pvalues[i])

            return dict_box_pairs, dict_pvalues

        return box_pairs, pvalues

    # def sig_brackets(
    #     self,
    #     stats_data: pd.DataFrame,
    #     stats_type: STATS_TYPE,
    #     plot_type: PLOT_TYPE,
    #     features: Optional[Union[str, Sequence[str]]] = None,
    #     subplots: Optional[bool] = False,
    # ):
    #     stats_data = StatsPipeline._filter_sig(stats_data)
    #
    #     # print(stats_data)
    #
    #     if features is None:
    #         features = []
    #     if isinstance(features, str):
    #         features = [features]
    #
    #     box_pairs = None
    #     pvals = None
    #     if stats_type == "mixed":
    #         # mixed
    #         stats_data = self._interaction_effect(stats_data)
    #         for col in _sig_cols:
    #             if col in stats_data.columns:
    #                 pvals = stats_data[col]
    #                 break
    #         stats_data = stats_data[[self.params["within"], "A", "B"]]
    #
    #         if plot_type == "multi":
    #             box_pairs = stats_data.apply(
    #                 # lambda grp: grp.apply(
    #                 lambda row: [(row.iloc[0], row["A"]), (row.iloc[0], row["B"])],
    #                 axis=1,
    #                 # )
    #             )
    #             box_pairs = [tuple(b) for b in box_pairs.values]
    #             pvals = pvals.values
    #
    #         print(box_pairs)
    #         print(pvals)
    #
    #     elif stats_type == "between":
    #         # between
    #         for col in _sig_cols:
    #             if col in stats_data.columns:
    #                 pvals = stats_data[col]
    #                 break
    #         stats_data = stats_data[["A", "B"]]
    #
    #         if plot_type == "multi":
    #             box_pairs = stats_data.reset_index(level=0).apply(
    #                 lambda row: [(row.iloc[0], row["A"]), (row.iloc[0], row["B"])],
    #                 axis=1,
    #             )
    #             box_pairs = list(box_pairs)
    #             box_pairs = [tuple(pair) for pair in box_pairs]
    #
    #     if plot_type == "multi":
    #         if len(features) > 0:
    #             box_pairs = [pair for pair in box_pairs if pair[0][0] in features]
    #             pvals = pd.concat([pvals.filter(like=f) for f in features])
    #
    #         if subplots:
    #             box_pairs, pvals = self.sig_brackets_dict(box_pairs, pvals)
    #
    #     elif plot_type == "single":
    #         # single
    #         if len(features) == 0:
    #             raise ValueError(
    #                 "Must specify `features` when `plot_type` is '{}'!".format(
    #                     plot_type
    #                 )
    #             )
    #
    #         print(stats_data)
    #         stats_data = stats_data.groupby(level=0).filter(
    #             lambda df: (df.reset_index().iloc[:, 0].isin(features)).all()
    #         )
    #         print(stats_data)
    #         box_pairs = [tuple(row) for row in stats_data[["A", "B"]].values]
    #
    #         pvals = pvals.groupby(pvals.reset_index().columns[0]).filter(
    #             lambda s: s.reset_index().iloc[:, 0].isin(features).all()
    #         )
    #
    #     return box_pairs, pvals

    def sig_brackets_dict(
        self,
        box_pairs: Union[
            Sequence[Tuple[Tuple[str, str], Tuple[str, str]]],
            Dict[str, Sequence[Tuple[Tuple[str, str], Tuple[str, str]]]],
        ],
        pvalues: pd.DataFrame,
    ) -> Tuple[
        Dict[str, Sequence[Tuple[Tuple[str, str], Tuple[str, str]]]],
        Dict[str, Sequence[float]],
    ]:
        dict_box_pairs = {}
        dict_pvalues = {}
        for pair, pval in zip(box_pairs, pvalues):
            print(pair, pval)
        # dict_pairs = {}
        #
        # if isinstance(sig_pairs, dict):
        #     dict_pairs = sig_pairs
        # else:
        #     for sig_pair in sig_pairs:
        #         pairs = dict_pairs.setdefault(sig_pair[0][0], [])
        #         pairs.append(sig_pair)
        #
        # dict_pvals = {
        #     key: (pvals.loc[key].values).flatten().tolist() for key in dict_pairs.keys()
        # }
        # return dict_pairs, dict_pvals

    def _param_df(self):
        return pd.DataFrame(
            [str(s) for s in self.params.values()],
            index=self.params.keys(),
            columns=["parameter"],
        )

    def _result_df(self):
        return pd.DataFrame(
            [s[1] for s in self.steps],
            index=[s[0] for s in self.steps],
            columns=["parameter"],
        )

    def df_to_latex(self, step: str, index_labels: Dict[str, str]):
        # TODO continue
        df = self.results[step]
        df = df[MAP_LATEX_EXPORT[step]]
        df.index = df.index.droplevel(-1)
        df = (
            df.rename(columns=MAP_LATEX)
            .reindex(index_labels.keys())
            .rename(index=index_labels)
        )
        return df

    def multicomp(
        self, stats_data: pd.DataFrame, method: Optional[str] = "bonf"
    ) -> pd.DataFrame:
        data = stats_data
        if stats_data.index.nlevels > 1:
            data = stats_data.groupby(list(stats_data.index.names)[:-1])
        return data.apply(lambda df: self._multicomp_lambda(df, method=method))

    def _multicomp_lambda(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        for col in list(reversed(_sig_cols[1:])):
            # iterate possible sig_cols in reserved order, except for 'p-corr'
            if col in data.columns:
                data["p-corr"] = pg.multicomp(list(data[col]), method=method)[1]
                break
        return data
