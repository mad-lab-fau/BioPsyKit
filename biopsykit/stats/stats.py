from typing import Dict, Sequence, Union, Tuple, Literal, Optional

import pandas as pd
import pingouin as pg
from IPython.core.display import display, Markdown
from biopsykit._types import path_t

MAP_STAT_TESTS = {
    'normality': pg.normality,
    'equal_var': pg.homoscedasticity,
    'anova': pg.anova,
    'welch_anova': pg.welch_anova,
    'rm_anova': pg.rm_anova,
    'mixed_anova': pg.mixed_anova,
    'kruskal': pg.kruskal,
    'pairwise_ttests': pg.pairwise_ttests,
    'pairwise_tukey': pg.pairwise_tukey
}

MAP_STAT_PARAMS = {
    'normality': ['dv', 'group'],
    'equal_var': ['dv', 'group'],
    'anova': ['dv', 'between'],
    'welch_anova': ['dv', 'between'],
    'rm_anova': ['dv', 'within', 'subject'],
    'mixed_anova': ['dv', 'between', 'within', 'subject'],
    'kruskal': ['dv', 'between'],
    'pairwise_ttests': ['dv', 'between', 'within', 'subject', 'effsize', 'tail', 'padjust'],
    'pairwise_tukey': ['dv', 'between', 'effsize']
}

MAP_NAMES = {
    'normality': "Test for Normal Distribution",
    'equal_var': "Test for Homoscedasticity (Equal Variances)",
    'anova': "ANOVA",
    'welch_anova': "Welch ANOVA",
    'rm_anova': "Repeated-measurement ANOVA",
    'mixed_anova': "Mixed ANOVA",
    'kruskal': "Kruskal-Wallis H-test for independent samples",
    'pairwise_ttests': "Pairwise t-Tests",
    'pairwise_tukey': "Pairwise Tukey's HSD (Honestly Significant Differences) Test"
}

MAP_CATEGORIES = {
    'prep': "Preparatory Analysis",
    'test': "Statistical Tests",
    'posthoc': "Post-Hoc Analysis"
}

MAP_LATEX_EXPORT = {
    'anova': ['ddof1', 'ddof2', 'F', 'p-unc', 'np2']
}

MAP_LATEX = {
    'ddof1': r"$\text{df}_{Num}$",
    'ddof2': r"$\text{df}_{Den}$",
    'F': 'F',
    'p-unc': 'p',
    'np2': r"$\eta^2_p$"
}

STATS_TYPE = Literal["within", "between", "mixed"]
PLOT_TYPE = Literal["single", "multi"]

_sig_cols = ['p-corr', 'p-tukey', 'p-unc', 'pval']


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

            general_params = {key: value for key, value in self.params.items() if len(key.split('__')) == 1}
            specific_params = {key.split('__')[1]: value for key, value in self.params.items() if
                               len(key.split('__')) > 1 and step[0] in key.split('__')}
            params = {key: general_params[key] for key in MAP_STAT_PARAMS[step[1]] if key in general_params}

            if 'groupby' in specific_params:
                grouper = specific_params.pop('groupby')
                result = data.groupby(grouper).apply(
                    lambda df: MAP_STAT_TESTS[step[1]](data=df, **specific_params, **params)
                )
            elif 'groupby' in general_params:
                grouper = general_params.pop('groupby')
                result = data.groupby(grouper).apply(
                    lambda df: MAP_STAT_TESTS[step[1]](data=df, **specific_params, **params)
                )
            else:
                result = MAP_STAT_TESTS[step[1]](data=data, **params)

            pipeline_results[step[1]] = result

        self.results = pipeline_results
        return pipeline_results

    def _ipython_display_(self):
        display(self._param_df().T)

    def display_results(self, **kwargs):
        sig_only = kwargs.get('sig_only', {})
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
                display(Markdown("""<font size="4"><b> {} </b></font>""".format(MAP_CATEGORIES[category])))
                for step in steps:
                    display(Markdown("**{}**".format(MAP_NAMES[step])))
                    df = self.results[step]
                    if sig_only.get(category, False):
                        df = self._filter_sig(df)
                        if df.empty:
                            display(Markdown("*No significant p-values.*"))
                            continue
                    display(df)

    @staticmethod
    def _filter_sig(df: pd.DataFrame) -> pd.DataFrame:
        for col in _sig_cols:
            if col in df.columns:
                return df[df[col] < 0.05]

    def export_statistics(self, file_path: path_t):
        writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
        workbook = writer.book
        header_format = workbook.add_format({'bold': True})
        param_df = self._param_df()
        param_df.to_excel(writer, sheet_name="parameter")
        for key, df in self.results.items():
            df.to_excel(writer, sheet_name=key, startrow=1)
            worksheet = writer.sheets[key]
            worksheet.write_string(0, 0, MAP_NAMES[key], header_format)
        writer.save()

    def sig_brackets(self, stats_data: pd.DataFrame, stats_type: STATS_TYPE, plot_type: PLOT_TYPE,
                     features: Optional[Union[str, Sequence[str]]] = None):
        stats_data = StatsPipeline._filter_sig(stats_data)

        if features is None:
            features = []
        if isinstance(features, str):
            features = [features]

        pvals = None
        if stats_type == 'mixed':
            stats_data = self._interaction_effect(stats_data)
            stats_data = stats_data[[self.params['within'], 'A', 'B']]
            for col in _sig_cols:
                if col in stats_data.columns:
                    pvals = stats_data[col]
                    break
        else:
            for col in _sig_cols:
                if col in stats_data.columns:
                    pvals = stats_data[col]
                    break

            stats_data = stats_data[['A', 'B']].reset_index(level=0)

        if plot_type == 'multi':
            box_pairs = list(stats_data.apply(lambda row: [(row.iloc[0], row['A']), (row.iloc[0], row['B'])], axis=1))
            box_pairs = [tuple(pair) for pair in box_pairs]
        elif plot_type == 'single':
            if len(features) == 0:
                raise ValueError("Must speficy `features` when `plot_type` is '{}'!".format(plot_type))
            stats_data = stats_data.groupby(stats_data.columns[0]).filter(
                lambda df: (df.iloc[:, 0].isin(features)).all()
            )
            box_pairs = [tuple(row) for row in stats_data[['A', 'B']].values]
            pvals = pvals.groupby(pvals.reset_index().columns[0]).filter(
                lambda s: s.reset_index().iloc[:, 0].isin(features).all()
            )
        else:
            box_pairs = None

        if plot_type == 'multi':
            box_pairs, pvals = self.sig_brackets_dict(box_pairs, pvals)
            if len(features) > 0:
                box_pairs = [box_pairs[f] for f in features]
                pvals = [pvals[f] for f in features]
                box_pairs = [x for pairs in box_pairs for x in pairs]
                pvals = [x for pval in pvals for x in pval]
        return box_pairs, pvals

    def sig_brackets_dict(self, sig_pairs: Sequence[Tuple[Tuple[str, str], Tuple[str, str]]],
                          pvals: pd.DataFrame) -> Tuple[
        Dict[str, Sequence[Tuple[Tuple[str, str], Tuple[str, str]]]], Dict[str, Sequence[float]]]:
        dict_pairs = {}

        for sig_pair in sig_pairs:
            pairs = dict_pairs.setdefault(sig_pair[0][0], [])
            pairs.append(sig_pair)

        dict_pvals = {key: (pvals.loc[key].values).flatten().tolist() for key in dict_pairs.keys()}
        return dict_pairs, dict_pvals

    def _param_df(self):
        return pd.DataFrame([str(s) for s in self.params.values()], index=self.params.keys(), columns=['parameter'])

    def _interaction_effect(self, stats_data: pd.DataFrame):
        return stats_data.loc[~stats_data[self.params['within']].eq('-')]

    def df_to_latex(self, step: str, index_labels: Dict[str, str]):
        # TODO continue
        df = self.results[step]
        df = df[MAP_LATEX_EXPORT[step]]
        df.index = df.index.droplevel(-1)
        df = df.rename(columns=MAP_LATEX).rename(index=index_labels)
        return df
