"""Module for setting up a pipeline for statistical analysis."""
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import pingouin as pg
from typing_extensions import Literal

from biopsykit.utils._datatype_validation_helper import _assert_file_extension, _assert_has_index_levels
from biopsykit.utils._types import path_t, str_t

MAP_STAT_TESTS = {
    "normality": pg.normality,
    "equal_var": pg.homoscedasticity,
    "anova": pg.anova,
    "welch_anova": pg.welch_anova,
    "rm_anova": pg.rm_anova,
    "mixed_anova": pg.mixed_anova,
    "ancova": pg.ancova,
    "kruskal": pg.kruskal,
    "friedman": pg.friedman,
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
    "ancova": ["dv", "between", "covar"],
    "kruskal": ["dv", "between"],
    "friedman": ["dv", "within", "subject"],
    "pairwise_ttests": ["dv", "between", "within", "subject", "effsize", "tail", "parametric", "padjust"],
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
    "ancova": "ANCOVA",
    "kruskal": "Kruskal-Wallis H-test for independent samples",
    "friedman": "Friedman test for repeated measurements",
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
    "T": "t",
    "T_collapse": r"$t({})$",
    "U-val": "U",
    "F_collapse": r"$F({}, {})$",
    "Q_collapse": r"$Q({})$",
    "dof": "df",
    "df1": "$df_{Den}$",
    "df2": "$df_{Nom}$",
    "p-unc": "p",
    "p-corr": "p",
    "p-tukey": "p",
    "pval": "p",
    "np2": r"$\eta^2_p$",
    "hedges": "Hedges' g",
}

STATS_CATEGORY = Literal["prep", "test", "posthoc"]
STATS_EFFECT_TYPE = Literal["between", "within", "interaction"]
PLOT_TYPE = Literal["single", "multi"]

_sig_cols = ["p-corr", "p-tukey", "p-unc", "pval"]


class StatsPipeline:
    """Class to set up a pipeline for statistical analysis."""

    def __init__(self, steps: Sequence[Tuple[str, str]], params: Dict[str, str], **kwargs):
        """Class to set up a pipeline for statistical analysis.

        The purpose of such a pipeline is to assemble several steps of a typical statistical analysis procedure while
        setting different parameters. The parameters passed to this class depend on the used statistical functions.
        It enables setting parameters of the various steps using their names and the parameter name separated by a "__",
        as in the examples below.

        The interface of this class is inspired by the scikit-learn Pipeline for machine learning tasks
        (:class:`~sklearn.pipeline.Pipeline`).

        All functions used are from the ``pingouin`` library (https://pingouin-stats.org/) for statistical analysis.

        The different steps of statistical analysis can be divided into categories:

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

        A ``StatsPipeline`` consists of a list of tuples specifying the individual ``steps`` of the pipeline.
        The first value of each tuple indicates the category this step belongs to (``prep``, ``test``, or ``posthoc``),
        the second value indicates the analysis function to use in this step (e.g., ``normality``, or ``rm_anova``).

        Furthermore, a ``params`` dictionary specifying the parameters and variables for statistical analysis
        needs to be supplied. Parameters can either be specified *globally*, i.e., for all steps in the pipeline
        (the default), or *locally*, i.e., only for one specific category, by prepending the category and separating it
        from the parameter name by a `__`. The parameters depend on the type of analysis used in the pipeline.

        Examples are:

        * ``dv``: column name of the dependent variable
        * ``between``: column name of the between-subject factor
        * ``within``: column name of the within-subject factor
        * ``effsize``: type of effect size to compute (if applicable)
        * ``multicomp``: whether (and how) to apply multi-comparison correction of p-values to the *last* step in the
          pipeline (either "test" or "posthoc") using :meth:`~biopsykit.stats.StatsPipeline.multicomp`.
          The arguments for the call to :meth:`~biopsykit.stats.StatsPipeline.multicomp` are supplied via dictionary.
        * ...

        Parameters
        ----------
        steps : list of tuples
            list of tuples specifying statistical analysis pipeline
        params : dict
            dictionary with parameter names and their values
        **kwargs : dict
            additional arguments, such as:

            * ``round``: Set the default decimal rounding of the output dataframes or ``None`` to disable rounding.
              Default: Rounding to 4 digits on p-value columns only. See :meth:`~pandas.DataFrame.round` for further
              options.

        """
        self.steps = steps
        self.params = params
        self.data: Optional[pd.DataFrame] = None
        self.results: Dict[str, pd.DataFrame] = {}
        self.category_steps = {}
        self.round = kwargs.get("round", {col: 4 for col in _sig_cols})
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

        for i, step in enumerate(self.steps):
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
                result = data.groupby(grouper, sort=general_params.get("sort", True)).apply(
                    lambda df: test_func(data=df, **specific_params, **params)  # pylint:disable=cell-var-from-loop
                )
            else:
                result = test_func(data=data, **specific_params, **params)

            if i == len(self.steps) - 1 and "multicomp" in general_params:
                # apply multi-comparison correction (p-adjustment) for the last analysis step in the pipeline
                # if it was enabled
                multicomp_dict = general_params.get("multicomp")
                if multicomp_dict is None:
                    multicomp_dict = {}
                result = self.multicomp(result, **multicomp_dict)

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
            from IPython.core.display import Markdown, display  # pylint:disable=import-outside-toplevel
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
                display(Markdown(f"""<font size="4"><b> {key} </b></font>"""))
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
            from IPython.core.display import Markdown, display  # pylint:disable=import-outside-toplevel
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
    def _filter_pcol(data: Union[pd.DataFrame, pd.Series]) -> Optional[pd.Series]:
        for col in _sig_cols:
            if isinstance(data, pd.DataFrame) and col in data.columns:
                return data[col]
            if isinstance(data, pd.Series) and col in data.index:
                return data[[col]]
        return None

    def _filter_effect(self, stats_category: STATS_CATEGORY, stats_effect_type: STATS_EFFECT_TYPE) -> pd.DataFrame:
        results = self.results_cat(stats_category)
        if len(results) == 0:
            raise ValueError(f"No results for category {stats_category}!")
        if "Contrast" in results.columns:
            if stats_effect_type == "interaction":
                key = f"{self.params['within']} * {self.params['between']}"
            else:
                key = self.params[stats_effect_type]

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
        stats_effect_type: STATS_EFFECT_TYPE,
        stats_type: Optional[STATS_EFFECT_TYPE] = None,
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
        stats_effect_type : {"between", "within", "interaction"}
            type of statistical effect ("between", "within", or "interaction"). Needed to extract the correct
            information from the analysis dataframe.
        stats_type : {"between", "within", "interaction"}
            .. note:: Deprecated in 0.4.0
                `stats_type` will be removed in 0.5.0, it is replaced by `stats_effect_type`.
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
        if stats_type is not None:
            warnings.warn(
                "Argument 'stats_type' is deprecated in 0.4.0 and was replaced by 'stats_effect_type'. "
                "It will be removed in 0.5.0.",
                category=DeprecationWarning,
            )
            stats_effect_type = stats_type

        stats_data = self._extract_stats_data(stats_category_or_data, stats_effect_type)

        if stats_effect_type == "interaction":
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

    def stats_to_latex(
        self,
        stats_test: Optional[str] = None,
        index: Optional[Union[Tuple, str]] = None,
        data: Optional[pd.Series] = None,
    ) -> str:
        """Generate LaTeX output from statistical results.

        Parameters
        ----------
        stats_test : str, optional
            name of statistical test in ``StatsPipeline``, e.g., "pairwise_ttests" or "anova" or ``None``
            if external statistical results is provided via ``data``
        index : str or tuple, optional
            row indexer of statistical result or ``None`` to generate LaTeX output for all rows.
            Default: ``None``
        data : :class:`~pandas.DataFrame`, optional
            dataframe with optional external statistical results

        Returns
        -------
        str
            LaTeX output that can be copied and pasted into LaTeX documents

        Raises
        ------
        ValueError
            if both ``data`` and ``stats_test`` is ``None``

        """
        if all(param is None for param in [stats_test, data]):
            raise ValueError("Either 'data' or 'stats_test' must be provided!")
        if stats_test is not None:
            data = self.results[stats_test].copy()
        if index is not None:
            data = data.loc[index].squeeze()

        if isinstance(data, pd.DataFrame):
            data = self._stats_to_index_set_index(data)
            return data.apply(self._stats_to_latex_row, axis=1)
        return self._stats_to_latex_row(data)

    def _stats_to_latex_row(self, row: pd.Series) -> str:
        pval = self._format_pval(row)
        if "T" in row:
            dof = self._format_dof(row["dof"])
            tval = self._format_number(row["T"])
            effsize = self._format_number(row["hedges"])
            return f"$t({dof}) = {tval}, p {pval}, g = {effsize}$"
        if "F" in row:
            rename_dict = {"ddof1": "df1", "ddof2": "df2", "DF": "df", "DF1": "df1", "DF2": "df2"}
            row = row.rename(rename_dict)
            dofs = (row["df1"], row["df2"]) if "df1" in row else (row["df"],)
            dofs = [self._format_dof(dof) for dof in dofs]
            dofs = ",".join(dofs)
            fval = self._format_number(row["F"])
            ret_string = rf"$F({dofs}) = {fval}, p {pval}$"
            if "np2" in row:
                effsize = self._format_number(row["np2"])
                ret_string = ret_string[:-1] + rf", \eta_p^2 = {effsize}$"
            return ret_string
        if "U-val" in row:
            effsize = self._format_number(row["hedges"])
            uval = self._format_number(row["U-val"])
            return f"$U = {uval}, p {pval}, g = {effsize}$"
        return ""

    def _format_pval(self, row: pd.Series) -> str:
        pval = round(self._filter_pcol(row)[0], 3)
        if pval < 0.001:
            pval = "< 0.001"
        elif pval > 0.999:
            pval = "> 0.999"
        else:
            pval = f"= {pval:.3f}"
        return pval

    def results_to_latex_table(  # pylint:disable=too-many-branches
        self,
        stats_test: str,
        data: Optional[pd.DataFrame] = None,
        stats_effect_type: Optional[STATS_EFFECT_TYPE] = None,
        unstack_levels: Optional[str_t] = None,
        collapse_dof: Optional[bool] = True,
        si_table_format: Optional[str] = None,
        index_kws: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        r"""Convert statistical result dataframe to LaTeX table.

        This function converts a dataframe from a statistical analysis to a LaTeX table using
        :meth:`~pandas.DataFrame.to_latex`.

        This function uses the LaTeX package ``siunitx`` (https://ctan.org/pkg/siunitx?lang=en) to represent numbers.
        By default, the column format for columns that contain numbers is "S" which is provided by ``siunitx``.
        The column format can be configured by the ``si_table_format`` argument.


        Parameters
        ----------
        stats_test : str, optional
            name of statistical test in ``StatsPipeline``, e.g., "pairwise_ttests" or "anova".
        data : :class:`~pandas.DataFrame`, optional
            dataframe with optional external statistical results
        stats_effect_type : {"between", "within", "interaction"}
            type of statistical effect ("between", "within", or "interaction"). Needed to extract the correct
            information from the analysis dataframe.
        unstack_levels : str or list of str, optional
            name(s) of dataframe index level(s) to be unstacked in the resulting latex table or ``None``
            to unstack no level(s)
        collapse_dof : bool, optional
            ``True`` to collapse degree-of-freedom (dof) from a separate column into the column header of the
            t- or F-value, respectively, ``False`` to keep it as separate "dof" column. This only works if
            the degrees-of-freedom are the same for all tests in the table. Default: ``True``
        si_table_format : str, optional
            table format for the numbers in the LaTeX table.
        index_kws : dict, optional
            dictionary containing arguments to configure how the table index is formatted. Possible arguments are:

            * index_italic : bool
              ``True`` to format index columns in italic, ``False`` otherwise. Default: ``True``
            * index_level_order : list
              list of index level names indicating the index level order of a :class:`~pandas.MultiIndex`
              in the LaTeX table. If `None` the index order of the dataframe will be used
            * index_value_order :  list or dict
              list of index values if rows in LaTeX table should have a different order than the underlying
              dataframe or if only specific rows should be exported as LaTeX table. If the table index is a
              :class:`~pandas.MultiIndex` then ``index_value_order`` should be a dictionary with the index level
              names as keys and lists of index values of the specific level as values
            * index_rename_map : dict
                mapping with dictionary with index values as keys and new index values to be exported
            * index_level_names_tex : str of list of str
                names of index levels in the LaTeX table or ``None`` to keep the index level names of the dataframe
        kwargs
            additional keywords that are passed to :meth:`~pandas.DataFrame.to_latex`.
            The following default arguments will be passed if not specified otherwise:

            * column_format: str
              The columns format as specified in LaTeX table format e.g. "rcl" for 3 columns. By default, the column
              format is automatically inferred from the dataframe, with index columns being formatted as "l" and
              value columns formatted as "S". If column headers are multi-columns, "|" will be added as separators
              between each group.
            * multicolumn_format : str
              The alignment for multi-columns. Default: "c"
            * escape : bool
              By default, the value will be read from the pandas config module. When set to ``False`` prevents from
              escaping latex special characters in column names. Default: ``False``
            * position : str
              The LaTeX positional argument for tables, to be placed after ``\begin{}`` in the output. Default: "th!"


        Returns
        -------
        str
            LaTeX code of formatted table

        """
        if data is None:
            data = self.results[stats_test].copy()

        if stats_effect_type is not None:
            data = data.set_index("Source", append=True)
            try:
                data = data.xs(stats_effect_type, level="Source")
            except KeyError:
                data = data.xs(self.params[stats_effect_type], level="Source")

        if si_table_format is None:
            si_table_format = "table-format = <1.3"

        if index_kws is None:
            index_kws = {}

        kwargs.setdefault("multicolumn_format", "c")
        kwargs.setdefault("escape", False)
        kwargs.setdefault("position", "th!")

        pcol = str(self._filter_pcol(data).name)
        data = self._extract_data_latex_table(data, stats_test=stats_test, pcol=pcol, collapse_dof=collapse_dof)

        column_map = {col: MAP_LATEX_EXPORT[col] for col in data.columns if col in MAP_LATEX_EXPORT}

        if len(data.index.names) > 1:
            data.index = data.index.droplevel(-1)
        data.loc[:, pcol] = data.loc[:, pcol].apply(self._format_pvals_stars)
        data = data.applymap(self._format_number)

        if unstack_levels is not None:
            data = data.stack()
            data = data.unstack(unstack_levels)
            data = data.unstack(-1)
        data = data.rename(columns=column_map)

        data = self._format_latex_table_index(data, index_kws)

        kwargs.setdefault("column_format", self._format_latex_column_format(data))

        data_latex = data.to_latex(**kwargs)
        return self._apply_latex_code_correction(data_latex, si_table_format)

    @staticmethod
    def _format_number(val):
        if isinstance(val, float):
            if val % 1 == 0:
                return str(int(val))
            return f"{val:.3f}"
        return val

    def multicomp(
        self,
        stats_category_or_data: Union[STATS_CATEGORY, pd.DataFrame],
        levels: Optional[Union[bool, str, Sequence[str]]] = False,
        method: Optional[str] = "bonf",
    ) -> pd.DataFrame:
        """Apply multi-comparison correction to results from statistical analysis.

        This function will add a new column ``p-corr`` to the dataframe which contains the adjusted p-values.
        The level(s) on which to perform multi-comparison correction on can be specified by the ``levels`` parameter.

        Parameters
        ----------
        stats_category_or_data : :class:`~pandas.DataFrame`
            dataframe with results from statistical analysis
        levels: bool, str, or list of str, optional
            index level(s) on which to perform multi-comparison correction on, ``True`` to perform multi-comparison
            correction on the *whole* dataset (i.e., on *no* particular index level), or ``False`` or ``None`` to
            perform multi-comparison correction on **all** index levels.
            Default: ``False``
        method : str, optional
            method used for testing and adjustment of p-values. See :func:`~pingouin.multicomp` for the
            available methods. Default: "bonf"

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with adjusted p-values

        """
        if isinstance(stats_category_or_data, pd.DataFrame):
            data = stats_category_or_data
        else:
            data = self.results_cat(stats_category_or_data)

        levels = self._multicomp_get_levels(levels, data)

        _assert_has_index_levels(data, levels, match_atleast=True)

        group_cols = list(data.index.names)[:-1]
        group_cols = list(set(group_cols) - set(levels))

        if len(group_cols) == 0:
            return self._multicomp_lambda(data, method=method)
        return data.groupby(group_cols).apply(lambda df: self._multicomp_lambda(df, method=method))

    @classmethod
    def _multicomp_get_levels(cls, levels: Union[bool, str, Sequence[str]], data: pd.DataFrame) -> Sequence[str]:
        if levels is None:
            levels = []
        elif isinstance(levels, bool):
            if levels:
                levels = list(data.index.names)[:-1]
            else:
                levels = []
        elif isinstance(levels, str):
            levels = [levels]
        return levels

    @staticmethod
    def _multicomp_lambda(data: pd.DataFrame, method: str) -> pd.DataFrame:
        for col in list(reversed(_sig_cols[1:])):
            # iterate possible sig_cols in reserved order, except for 'p-corr'
            if col in data.columns:
                data["p-corr"] = pg.multicomp(list(data[col]), method=method)[1]
                break
        return data

    def _extract_stats_data(
        self, stats_category_or_data: Union[STATS_CATEGORY, pd.DataFrame], stats_effect_type: STATS_EFFECT_TYPE
    ):
        if isinstance(stats_category_or_data, (str, pd.DataFrame)):
            if isinstance(stats_category_or_data, str):
                stats_data = self._filter_effect(stats_category_or_data, stats_effect_type)
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

        stats_data = stats_data.drop_duplicates()
        stats_data = stats_data.reset_index()
        if plot_type == "single":
            box_pairs = self._get_box_pairs_single(stats_data)
        else:
            box_pairs = self._get_box_pairs_multi(stats_data, x)

        return stats_data, box_pairs

    def _get_stats_data_box_pairs_interaction(self, stats_data: pd.DataFrame):
        stats_data = stats_data.reset_index()
        stats_data = stats_data[stats_data[self.params["within"]] != "-"]
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

    def _get_box_pairs_multi(self, stats_data: pd.DataFrame, x: str):
        if x is None:
            raise ValueError("'x' must be specified when 'plot_type' is 'multi'!")
        if x in stats_data.columns:
            stats_data = stats_data.set_index(x)
        else:
            stats_data = stats_data.set_index(self.params["groupby"])
        return stats_data.apply(lambda row: ((row.name, row["A"]), (row.name, row["B"])), axis=1)

    def _display_category(  # pylint:disable=too-many-branches
        self, category: str, steps: Sequence[str], sig_only: Dict[str, bool], groupby: str, group_key: str
    ):
        try:
            from IPython.core.display import Markdown, display  # pylint:disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError(
                "Displaying statistics results failed because "
                "IPython cannot be imported. Install it via 'pip install ipython'."
            ) from e
        display(Markdown(f"""<font size="3"><b> {MAP_CATEGORIES[category]} </b></font>"""))
        for step in steps:
            display(Markdown(f"**{MAP_NAMES[step]}**"))
            df = self.results[step]
            if groupby is not None:
                df = df.xs(group_key, level=groupby)
            if sig_only.get(category, False):
                df = self._filter_sig(df)
                if df.empty:
                    display(Markdown("*No significant p-values.*"))
                    continue
            if self.round is None:
                display(df)
            else:
                display(df.round(self.round))

    @staticmethod
    def _format_pvals_stars(pval: float) -> str:
        pstar = pd.cut([pval], [0.0, 0.001, 0.01, 0.05, 1.1], right=False, labels=["***", "**", "*", ""])
        pstar = pstar[0]
        ret = f"{pval:.3f}"
        if pval == 1.0:
            ret = ">0.999"
        if len(pstar) >= 0:
            if pval < 0.001:
                ret = "<0.001"
            ret += rf"$^{{{pstar}}}$"
        return ret

    @staticmethod
    def _format_dof(dof: int) -> str:
        return str(int(dof)) if dof % 1 == 0 else f"{dof:.2f}"

    def _extract_data_ttest(self, data: pd.DataFrame, pcol: str, collapse_dof: bool) -> pd.DataFrame:
        columns = ["T", "dof", pcol, "hedges"]
        data = data[columns]
        if collapse_dof:
            dof = data["dof"].unique()
            if len(dof) != 1:
                raise ValueError(f"Cannot collapse dof in table: dof are not unique! Got {dof}")
            data = data.rename(columns={"T": MAP_LATEX_EXPORT["T_collapse"].format(self._format_dof(dof[0]))})
            data = data.drop(columns="dof")
        return data

    def _extract_data_anova(self, data: pd.DataFrame, pcol: str, collapse_dof: bool) -> pd.DataFrame:
        rename_dict = {"ddof1": "df1", "ddof2": "df2", "DF": "df", "DF1": "df1", "DF2": "df2"}
        data = data.rename(columns=rename_dict)
        columns = []
        if collapse_dof:
            if "df1" in data.columns:
                dof_cols = ["df1", "df2"]
            else:
                dof_cols = ["df"]
            dofs = tuple((data[col].unique() for col in dof_cols))
            if any(len(d) != 1 for d in dofs):
                raise ValueError(f"Cannot collapse dof in table: dof are not unique! Got {dofs}.")
            dofs = [self._format_dof(dof[0]) for dof in dofs]
            f_col = MAP_LATEX_EXPORT["F_collapse"].format(*dofs)
            columns.append(f_col)
            data = data.rename(columns={"F": f_col})
            data = data.drop(columns=["df1", "df2", "df"], errors="ignore")
        else:
            if "df1" in data.columns:
                data["df"] = "{" + data["df1"].astype(str) + ", " + data["df2"].astype(str) + "}"
                data = data.drop(columns=["df1", "df2"])
            columns.append("df")
        columns = columns + [pcol, "np2"]
        data = data[columns]
        return data

    def _extract_data_friedman(self, data: pd.DataFrame, pcol: str, collapse_dof: bool) -> pd.DataFrame:
        rename_dict = {"ddof1": "df1", "ddof2": "df2"}
        data = data.rename(columns=rename_dict)
        columns = []
        if collapse_dof:
            if "df2" in data.columns:
                dof_cols = ["df1", "df2"]
            else:
                dof_cols = ["df1"]
            dofs = tuple((data[col].unique() for col in dof_cols))
            if any(len(d) != 1 for d in dofs):
                raise ValueError(f"Cannot collapse dof in table: dof are not unique! Got {dofs}.")
            dofs = [self._format_dof(dof[0]) for dof in dofs]
            if "F" in data.columns:
                f_col = MAP_LATEX_EXPORT["F_collapse"].format(*dofs)
                columns.append(f_col)
                data = data.rename(columns={"F": f_col})
                data = data.drop(columns=["df1", "df2", "df"], errors="ignore")
            else:
                q_col = MAP_LATEX_EXPORT["Q_collapse"].format(*dofs)
                columns.append(q_col)
                data = data.rename(columns={"Q": q_col})
                data = data.drop(columns=["df1", "df2", "df"], errors="ignore")
        else:
            if "df2" in data.columns:
                data["df"] = "{" + data["df1"].astype(str) + ", " + data["df2"].astype(str) + "}"
                data = data.drop(columns=["df1", "df2"])
            columns.append("df")
        columns = columns + [pcol]
        data = data[columns]
        return data

    @staticmethod
    def _extract_data_mwu(data: pd.DataFrame, pcol: str) -> pd.DataFrame:
        columns = ["U-val", pcol, "hedges"]
        data = data[columns]
        return data

    @staticmethod
    def _format_latex_table_index(data, index_kws):  # pylint:disable=too-many-branches
        index_italic = index_kws.get("index_italic", True)
        index_level_order = index_kws.get("index_level_order", None)
        index_value_order = index_kws.get("index_value_order", None)
        index_rename_map = index_kws.get("index_rename_map", None)
        index_level_names_tex = index_kws.get("index_level_names_tex", None)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns.names = [None for _ in data.columns.names]
        else:
            data.columns.name = None

        if index_level_order is not None:
            data = data.reorder_levels(index_level_order)

        if index_value_order is not None:
            data = StatsPipeline._apply_index_value_order(data, index_value_order)

        if index_rename_map is not None:
            data = data.rename(index=index_rename_map)
        if index_level_names_tex is not None:
            if isinstance(index_level_names_tex, str):
                index_level_names_tex = [index_level_names_tex]
            data.index.names = index_level_names_tex

        data = data.add_prefix("{").add_suffix("}")

        if index_italic:
            data.index.names = [rf"\textit{{{s}}}" for s in data.index.names]
            data = data.T
            data = data.add_prefix(r"\textit{").add_suffix("}")
            data = data.T
        return data

    @staticmethod
    def _apply_index_value_order(
        data: pd.DataFrame, index_value_order: Union[Sequence[str], Dict[str, Sequence[str]]]
    ) -> pd.DataFrame:
        if isinstance(data.index, pd.MultiIndex):
            if isinstance(index_value_order, dict):
                for key, val in index_value_order.items():
                    data = data.reindex(val, level=key)
            else:
                raise ValueError(
                    "'index_value_order' must be a dictionary with index level names as keys and "
                    "index values as values."
                )
        else:
            if isinstance(index_value_order, dict):
                data = data.reindex(index_value_order[data.index.name])
            else:
                data = data.reindex(index_value_order)
        return data

    @staticmethod
    def _format_latex_column_format(data: pd.DataFrame):
        column_format = "l" * data.index.nlevels + "||"
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
    def _apply_latex_code_correction(data_latex: str, si_table_format: str):
        data_latex = data_latex.replace(r"\textasciicircum ", "^")
        if si_table_format is not None:
            data_latex = re.sub(r"(\\begin\{tabular\})", r"\\sisetup{" + si_table_format + r"}\n\n\1", data_latex)
        return data_latex

    @staticmethod
    def _stats_to_index_set_index(data: pd.DataFrame) -> pd.DataFrame:
        if "Contrast" in data.columns:
            data = data.set_index("Contrast", append=True)
        if "Source" in data.columns:
            data = data.set_index("Source", append=True)
        if "A" in data.columns:
            data = data.set_index(["A", "B"], append=True)
        return data

    def _extract_data_latex_table(
        self, data: pd.DataFrame, stats_test: str, pcol: str, collapse_dof: bool
    ) -> pd.DataFrame:
        if "anova" in stats_test:
            data = self._extract_data_anova(data, pcol, collapse_dof)
        if "friedman" in stats_test:
            data = self._extract_data_friedman(data, pcol, collapse_dof)
        if "T" in stats_test:
            data = self._extract_data_ttest(data, pcol, collapse_dof)
        if "U-val" in data.columns:
            data = self._extract_data_mwu(data, pcol)

        return data
