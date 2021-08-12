"""Module for setting up a pipeline for statistical analysis."""
from pathlib import Path
from typing import Dict, Sequence, Union, Tuple, Optional, List
from typing_extensions import Literal

import pandas as pd
import pingouin as pg

from biopsykit.utils._datatype_validation_helper import _assert_file_extension
from biopsykit.utils._types import path_t

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
    "pairwise_ttests": ["dv", "between", "within", "subject", "effsize", "tail", "padjust", "parametric"],
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

STATS_CATEGORY = Literal["prep", "test", "posthoc"]
STATS_TYPE = Literal["between", "within", "interaction"]
PLOT_TYPE = Literal["single", "multi"]

_sig_cols = ["p-corr", "p-tukey", "p-unc", "pval"]


class StatsPipeline:
    """Class to set up a pipeline for statistical analysis."""

    def __init__(self, steps: Sequence[Tuple[str, str]], params: Dict[str, str]):
        """Class to set up a pipeline for statistical analysis.

        The purpose of such a pipeline is to assemble several steps of a typical statistical analysis procedure while
        setting different parameters. The parameters passed to this class depend on the used statistical functions.
        It enables setting parameters of the various steps using their names and the parameter name separated by a "__",
        as in the examples below.

        The interface of this class is inspired by the scikit-learn Pipeline for ML tasks
        (:class:`~sklearn.pipeline.Pipeline`).

        All functions methods used are from the Pingouin library (https://pingouin-stats.org/) for statistical analysis.

        The different steps of statistical analysis is divided into different categories:

        * *Preparatory Analysis* (``prep``): Analyses applied to the data before performing the actual statistical
          analysis. Currently supported functions are:

          * ``normality``: Test whether a random sample comes from a normal distribution.
            See :func:`~pingouin.normality` for further information.
          * ``equal_var``: Test equality of variances (homoscedasticity).
            See :func:`~pingouin.homoscedasticity` for further information.

        * *Statistical Test* (``test``): Statistical test to determine differences or similarities in the data.
          Currently supported functions are:

            * ``pairwise_ttests``: Pairwise T-tests (either for independent or dependent samples).
              See :func:`~pingouin.pairwise_ttests` for further information.
            * ``anova``: One-way or N-way ANOVA. See :func:`~pingouin.anova` for further information.
            * ``welch_anova``: One-way Welch-ANOVA. See :func:`~pingouin.welch_anova` for further
              information.
            * ``rm_anova``: One-way and two-way repeated measures ANOVA. See :func:`~pingouin.rm_anova`
              further information.
            * ``mixed_anova``: Mixed-design (split-plot) ANOVA. See :func:`~pingouin.mixed_anova` for
              further information.
            * ``kruskal``: Kruskal-Wallis H-test for independent samples. See :func:`~pingouin.kruskal`
              for further information.

        * *Posthoc Tests* (``posthoc``): Posthoc tests to determine differences of individual groups if more than two
          groups are analyzed.
          Currently supported functions are:

          * ``pairwise_ttests``: Pairwise T-tests (either for independent or dependent samples).
            See :func:`~pingouin.pairwise_ttests` for further information.
          * ``pairwise_tukey``: Pairwise Tukey-HSD post-hoc test.
            See :func:`~pingouin.pairwise_tukey` for further information.
          * ``pairwise_gameshowell``: Pairwise Games-Howell post-hoc test.
            See :func:`~pingouin.pairwise_gameshowell` for further information.

        Initialize new ``StatsPipeline`` instance.

        A ``StatsPipeline`` consists of a list of tuples specifying the individual ``steps`` of the pipeline.
        The first value of each tuple indicates the category this step belongs to (``prep``, ``test``, or ``posthoc``),
        the second value indicates the analysis function to use in this step (e.g., ``normality``, or ``rm_anova``).

        Furthermore, a ``params`` dictionary specifying the parameters and variables for statistical analysis
        needs to be supplied. Parameters can either be specified *globally*, i.e., for all steps in the pipeline
        (the default), or *locally*, i.e., only for one specific category, by prepending the category,
        separated by a "__". The parameters depend on the type of analysis used in the pipeline. Examples are:

        * ``dv``: column name of the dependent variable
        * ``between``: column name of the between-subject factor
        * ``within``: column name of the within-subject factor
        * ``effsize``: type of effect size to compute (if applicable)
        * ...

        Parameters
        ----------
        steps : list of tuples
            list of tuples specifying statistical analysis pipeline
        params : dict
            dictionary with parameter names and their values

        """
        self.steps = steps
        self.params = params
        self.data: Optional[pd.DataFrame] = None
        self.results: Dict[str, pd.DataFrame] = {}
        self.category_steps = {}
        for step in self.steps:
            self.category_steps.setdefault(step[0], [])
            self.category_steps[step[0]].append(step[1])

    def apply(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Apply statistical analysis pipeline on input data.

        Parameters
        ----------
        data : :class:`~pandas.DataFrame`
            input data to apply statistical analysis pipeline on. Must be provided in *long-format*.

        Returns
        -------
        dict
            dictionary with results from all pipeline steps

        """
        self.data = data
        pipeline_results = {}
        data = data.reset_index()

        for step in self.steps:
            general_params = {key: value for key, value in self.params.items() if len(key.split("__")) == 1}
            specific_params = {
                key.split("__")[1]: value
                for key, value in self.params.items()
                if len(key.split("__")) > 1 and step[0] in key.split("__")
            }
            params = {key: general_params[key] for key in MAP_STAT_PARAMS[step[1]] if key in general_params}

            grouper = []
            grouper_tmp = self._get_grouper_variable(general_params, specific_params)
            grouper = grouper + grouper_tmp

            if step[0] == "prep":
                grouper, specific_params = self._get_specific_params_prep(grouper, general_params, specific_params)

            test_func = MAP_STAT_TESTS[step[1]]
            if len(grouper) > 0:
                result = data.groupby(grouper).apply(
                    lambda df: test_func(data=df, **specific_params, **params)  # pylint:disable=cell-var-from-loop
                )
            else:
                result = test_func(data=data, **specific_params, **params)

            if step[0] == "posthoc" and "padjust" in general_params and "padjust" not in params:
                # apply p-adjustment for posthoc testing if it was specified in the pipeline
                # but do it only manually if it's not supported by the test function
                # (otherwise it would be in the 'params' dict)
                result = self.multicomp(result, method=general_params["padjust"])

            pipeline_results[step[1]] = result

        self.results = pipeline_results
        return pipeline_results

    @staticmethod
    def _get_grouper_variable(general_params: Dict[str, str], specific_params: Dict[str, str]):
        grouper_tmp = []
        if "groupby" in specific_params:
            grouper_tmp = specific_params.pop("groupby")
        elif "groupby" in general_params:
            grouper_tmp = general_params.pop("groupby")

        if isinstance(grouper_tmp, str):
            grouper_tmp = [grouper_tmp]

        return grouper_tmp

    @staticmethod
    def _get_specific_params_prep(grouper: List[str], general_params: Dict[str, str], specific_params: Dict[str, str]):
        if "within" in general_params and "between" in general_params:
            grouper.append(general_params["within"])
            specific_params["group"] = general_params["between"]
        else:
            specific_params["group"] = general_params.get("within", general_params.get("between"))

        return grouper, specific_params

    def results_cat(self, category: STATS_CATEGORY) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Return results for pipeline category.

        This function filters results from steps belonging to the specified category and returns them.

        Parameters
        ----------
        category : {"prep", "test", "posthoc"}
            category name

        Returns
        -------
        :class:`~pandas.DataFrame` or dict
            dataframe with results from the specified category or dict of such if multiple steps belong to the
            same category.

        """
        cat = self.category_steps.get(category, [])
        if len(cat) == 1:
            return self.results[cat[0]]
        if len(cat) > 1:
            return {c: self.results[c] for c in cat}
        return {}

    def _ipython_display_(self):
        try:
            from IPython.core.display import display  # pylint:disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError(
                "Displaying statistics results failed because "
                "IPython cannot be imported. Install it via 'pip install ipython'."
            ) from e

        display(self._param_df().T)
        display(self._result_df().T)

    def display_results(self, sig_only: Optional[Union[str, bool, Sequence[str], Dict[str, bool]]] = None, **kwargs):
        """Display formatted results of statistical analysis pipeline.

        This function displays the results of the statistical analysis pipeline. The output is Markdown-formatted and
        optimized for Jupyter Notebooks. The output can be configured to for example:

        * only show specific categories
        * only show statistically significant results for specific categories or for all pipeline categories
        * display results grouped by a grouper (when pipeline is applied on multiple groups of data, e.g.,
          on multiple feature of the same type independently)

        Parameters
        ----------
        sig_only : bool, str, list, or dict, optional
            whether to only show statistically significant (p < 0.05) results or not. ``sig_only`` accepts multiple
            possible formats:

            * ``str``: filter only one specific category or "all" to filter all categories by statistical
              significance
            * ``bool``: ``True`` to filter all categories by statistical significance, ``False`` otherwise
            * ``list``: list of categories whose results should be filtered by statistical significance
            * ``dict``: dictionary with category names and bool values to filter (or not) for statistical
              significance

            Default: ``None`` (no filtering)
        **kwargs
            additional arguments to be passed to the function, such as:

            * ``category`` names: ``True`` to display results of this category, ``False`` to skip displaying results
              of this category. Default: show results from all categories
            * ``grouped``: ``True`` to group results by the variable "groupby" specified in the parameter
              dictionary when initializing the ``StatsPipeline`` instance.

        """
        try:
            from IPython.core.display import display, Markdown  # pylint:disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError(
                "Displaying statistics results failed because "
                "IPython cannot be imported. Install it via 'pip install ipython'."
            ) from e

        sig_only = self._get_sig_only(sig_only)
        grouped = kwargs.pop("grouped", False)

        if self.results is None:
            display(Markdown("No results."))
            return

        if grouped and "groupby" in self.params:
            for key, _ in self.data.groupby(self.params.get("groupby")):
                display(Markdown("""<font size="4"><b> {} </b></font>""".format(key)))
                self._display_results(sig_only, self.params.get("groupby"), key, **kwargs)
        else:
            self._display_results(sig_only, **kwargs)

    def _get_sig_only(self, sig_only: Optional[Union[str, bool, Sequence[str], Dict[str, bool]]] = None):
        if sig_only is None:
            sig_only = {}
        if isinstance(sig_only, str):
            if sig_only == "all":
                sig_only = True
            else:
                sig_only = [sig_only]
        if isinstance(sig_only, bool):
            sig_only = {cat: sig_only for cat in self.category_steps}
        if isinstance(sig_only, list):
            sig_only = {cat: cat in sig_only for cat in self.category_steps}
        return sig_only

    def _display_results(
        self, sig_only: Dict[str, bool], groupby: Optional[str] = None, group_key: Optional[str] = None, **kwargs
    ):
        try:
            from IPython.core.display import display, Markdown  # pylint:disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError(
                "Displaying statistics results failed because "
                "IPython cannot be imported. Install it via 'pip install ipython'."
            ) from e

        display(Markdown("""<font size="3"><b> Overview </b></font>"""))
        display(self)
        for category, steps in self.category_steps.items():
            if kwargs.get(category, True):
                self._display_category(category, steps, sig_only, groupby, group_key)

    @staticmethod
    def _filter_sig(data: pd.DataFrame) -> Optional[pd.DataFrame]:
        for col in _sig_cols:
            if col in data.columns:
                if data[col].isna().all():
                    # drop column if all values are NaN => most probably because we turned on p-adjust but only
                    # have two main effects
                    data = data.drop(columns=col)
                    continue
                return data[data[col] < 0.05]
        return None

    @staticmethod
    def _filter_pcol(data: pd.DataFrame) -> Optional[pd.Series]:
        for col in _sig_cols:
            if col in data.columns:
                return data[col]
        return None

    def _filter_effect(self, stats_category: STATS_CATEGORY, stats_type: STATS_TYPE) -> pd.DataFrame:
        results = self.results_cat(stats_category)
        if len(results) == 0:
            raise ValueError("No results for category {}!".format(stats_category))
        if "Contrast" in results.columns:
            if stats_type == "interaction":
                key = "{} * {}".format(self.params["within"], self.params["between"])
            else:
                key = self.params[stats_type]

            results = results[results["Contrast"] == key]
            results = results.drop(columns="Contrast")

        return results

    def export_statistics(self, file_path: path_t):
        """Export results of statistics analysis pipeline to Excel file.

        Each step of the analysis pipeline is saved into its own sheet. The first sheet is an overview of the
        parameters specified in the analysis pipeline.

        Parameters
        ----------
        file_path : :class:`~pathlib.Path` or str
            path to export file

        """
        # assert pathlib
        file_path = Path(file_path)
        _assert_file_extension(file_path, ".xlsx")

        writer = pd.ExcelWriter(file_path, engine="xlsxwriter")  # pylint:disable=abstract-class-instantiated
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
        stats_category_or_data: Union[STATS_CATEGORY, pd.DataFrame],
        stats_type: STATS_TYPE,
        plot_type: Optional[PLOT_TYPE] = "single",
        features: Optional[Union[str, Sequence[str], Dict[str, Union[str, Sequence[str]]]]] = None,
        x: Optional[str] = None,
        subplots: Optional[bool] = False,
    ) -> Union[
        Tuple[Sequence[Tuple[str, str]], Sequence[float]],
        Tuple[Dict[str, Sequence[Tuple[Tuple[str, str], Tuple[str, str]]]], Dict[str, Sequence[float]]],
    ]:
        """Generate *significance brackets* used indicate statistical significance in boxplots.

        Parameters
        ----------
        stats_category_or_data : {"prep", "test", "posthoc"} or :class:`~pandas.DataFrame`
            either a string to specify the pipeline category to use for generating significance brackets or a
            dataframe with statistical results if significance brackets should be generated from the dataframe.
        stats_type : {"between", "within", "interaction"}
            type of analysis performed ("between", "within", or "interaction"). Needed to extract the correct
            information from the analysis dataframe.
        plot_type : {"single", "multi"}
            type of plot for which significance brackets are generated: "multi" if boxplots are grouped
            (by ``hue`` variable), "single" (the default) otherwise.
        features : str, list or dict, optional
            feature(s) used in boxplot. The resulting significance brackets will be filtered accordingly to only
            contain features present in the boxplot. It can have the following formats:

            * ``str``: only one feature is plotted in the boxplot
              (returns significance brackets of only one feature)
            * ``list``: multiple features are combined into *one* :class:`~matplotlib.axes.Axes` object
              (returns significance brackets of multiple features)
            * ``dict``: dictionary with feature (or list of features) per subplot if boxplots are structured in
              subplots (``subplots`` is ``True``) (returns dictionary with significance brackets per subplot)

            Default: ``None`` to return significance brackets of all features
        x : str, optional
            name of ``x`` variable used to plot data in boxplot. Only required if ``plot_type`` is "multi".
        subplots : bool, optional
            ``True`` if multiple boxplots are structured in subplots, ``False`` otherwise. Default: ``False``

        Returns
        -------
        box_pairs
            list with significance brackets (or dict of such if ``subplots`` is ``True``)
        pvalues
            list with p values belonging to the significance brackets in ``box_pairs`` (or dict of such if
            ``subplots`` is ``True``)

        """
        features = self._sanitize_features_input(features)

        stats_data = self._extract_stats_data(stats_category_or_data, stats_type)

        if stats_type == "interaction":
            stats_data, box_pairs = self._get_stats_data_box_pairs_interaction(stats_data)
        else:
            stats_data, box_pairs = self._get_stats_data_box_pairs(stats_data, plot_type, features, x)

        if box_pairs.empty:
            return [], []

        pvalues = self._filter_pcol(stats_data)

        if subplots:
            return self._sig_brackets_dict(box_pairs, pvalues, features)

        return list(box_pairs), list(pvalues)

    @staticmethod
    def _sig_brackets_dict(
        box_pairs: pd.Series,
        pvalues: pd.Series,
        features: Union[Sequence, Dict[str, Union[str, Sequence[str]]]],
    ) -> Tuple[Dict[str, Sequence[Tuple[Tuple[str, str], Tuple[str, str]]]], Dict[str, Sequence[float]]]:
        dict_box_pairs = {}
        dict_pvalues = {}
        if features is None:
            features = list(box_pairs.index.unique())
        if isinstance(features, list):
            features = {f: f for f in features}
        for key in features:
            features_list = features[key]
            if isinstance(features_list, str):
                features_list = [features_list]

            list_pairs = dict_box_pairs.setdefault(key, [])
            list_pvalues = dict_pvalues.setdefault(key, [])
            for i, (idx, sig_pair) in enumerate(box_pairs.iteritems()):
                if idx in features_list:
                    list_pairs.append(sig_pair)
                    list_pvalues.append(pvalues[i])

        return dict_box_pairs, dict_pvalues

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
        """Convert result dataframe to prepare for LaTeX export.

        This function converts a dataframe from an analysis step of the statistics pipeline to prepare it for the
        export as LaTeX code using :meth:`~pandas.DataFrame.to_latex`.

        Parameters
        ----------
        step : str
            step of statistical analysis pipeline
        index_labels :
            dictionary to rename index labels

        Returns
        -------
        :class:`~pandas.DataFrame`
            converted dataframe

        """
        # TODO continue
        df = self.results[step]
        df = df[MAP_LATEX_EXPORT[step]]
        df.index = df.index.droplevel(-1)
        df = df.rename(columns=MAP_LATEX).reindex(index_labels.keys()).rename(index=index_labels)
        return df

    def multicomp(self, stats_data: pd.DataFrame, method: Optional[str] = "bonf") -> pd.DataFrame:
        """Apply multi-test comparison to results from statistical analysis.

        This function will add a new column ``p-corr`` to the dataframe which contains the adjusted p values.

        Parameters
        ----------
        stats_data : :class:`~pandas.DataFrame`
            dataframe with results from statistical analysis
        method : str, optional
            method used for testing and adjustment of p-values. See :func:`~pingouin.multicomp` for the
            available methods. Default: "bonf"

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with adjusted p values

        """
        data = stats_data
        if stats_data.index.nlevels > 1:
            data = stats_data.groupby(list(stats_data.index.names)[:-1])
            return data.apply(lambda df: self._multicomp_lambda(df, method=method))
        return self._multicomp_lambda(data, method=method)

    @staticmethod
    def _multicomp_lambda(data: pd.DataFrame, method: str) -> pd.DataFrame:
        for col in list(reversed(_sig_cols[1:])):
            # iterate possible sig_cols in reserved order, except for 'p-corr'
            if col in data.columns:
                data["p-corr"] = pg.multicomp(list(data[col]), method=method)[1]
                break
        return data

    def _extract_stats_data(self, stats_category_or_data: Union[STATS_CATEGORY, pd.DataFrame], stats_type: STATS_TYPE):
        if isinstance(stats_category_or_data, (str, pd.DataFrame)):
            if isinstance(stats_category_or_data, str):
                stats_data = self._filter_effect(stats_category_or_data, stats_type)
            else:
                stats_data = stats_category_or_data
        else:
            raise ValueError(
                "Either string with stats category (e.g., 'test' or 'posthoc') or dataframe with stats results must "
                "be supplied as parameter! "
            )
        return self._filter_sig(stats_data)

    @staticmethod
    def _sanitize_features_input(features: Union[str, Sequence[str], Dict[str, Union[str, Sequence[str]]]]):
        if isinstance(features, str):
            features = [features]
        if isinstance(features, dict):
            for key, value in features.items():
                # ensure that all entries in the dict are lists for consistency
                if isinstance(value, str):
                    features[key] = [value]

        return features

    def _get_stats_data_box_pairs(
        self,
        stats_data: pd.DataFrame,
        plot_type: Optional[PLOT_TYPE] = "single",
        features: Optional[Union[Sequence[str], Dict[str, Union[str, Sequence[str]]]]] = None,
        x: Optional[str] = None,
    ):
        if features is not None:
            if isinstance(features, dict):
                # flatten dict values into list of str
                features = list(features.values())
                features = [item for sublist in features for item in sublist]

            stats_data = pd.concat([stats_data.filter(like=f, axis=0) for f in features])

        stats_data = stats_data.reset_index()
        if plot_type == "single":
            box_pairs = self._get_box_pairs_single(stats_data)
        else:
            box_pairs = self._get_box_pairs_multi(stats_data, x)

        return stats_data, box_pairs

    def _get_stats_data_box_pairs_interaction(self, stats_data: pd.DataFrame):
        stats_data = stats_data.reset_index()
        index = stats_data[self.params.get("groupby", [])]
        stats_data = stats_data.set_index(self.params["within"])

        box_pairs = stats_data.apply(lambda row: ((row.name, row["A"]), (row.name, row["B"])), axis=1)
        if not index.empty:
            box_pairs.index = index
        return stats_data, box_pairs

    def _get_box_pairs_single(self, stats_data: pd.DataFrame):
        try:
            box_pairs = stats_data.apply(lambda row: (row["A"], row["B"]), axis=1)
        except KeyError as e:
            raise ValueError(
                "Generating significance brackets failed. If ANOVA (or such) was used as "
                "statistical test, significance brackets need to be generated from post-hoc tests!"
            ) from e
        index = stats_data[self.params.get("groupby", [])]
        if not index.empty:
            box_pairs.index = index
        return box_pairs

    @staticmethod
    def _get_box_pairs_multi(stats_data: pd.DataFrame, x: str):
        if x is None:
            raise ValueError("'x' must be specified when 'plot_type' is 'multi'!")
        stats_data = stats_data.set_index(x)
        return stats_data.apply(lambda row: ((row.name, row["A"]), (row.name, row["B"])), axis=1)

    def _display_category(
        self, category: str, steps: Sequence[str], sig_only: Dict[str, bool], groupby: str, group_key: str
    ):
        try:
            from IPython.core.display import display, Markdown  # pylint:disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError(
                "Displaying statistics results failed because "
                "IPython cannot be imported. Install it via 'pip install ipython'."
            ) from e
        display(Markdown("""<font size="3"><b> {} </b></font>""".format(MAP_CATEGORIES[category])))
        for step in steps:
            display(Markdown("**{}**".format(MAP_NAMES[step])))
            df = self.results[step]
            if groupby is not None:
                df = df.xs(group_key, level=groupby)
            if sig_only.get(category, False):
                df = self._filter_sig(df)
                if df.empty:
                    display(Markdown("*No significant p-values.*"))
                    continue
            display(df)
