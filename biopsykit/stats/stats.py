from typing import Dict, Sequence, Union, Tuple

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


class StatsPipeline:

    def __init__(self, steps: Sequence[Tuple[str, str]], params: Dict[str, str]):
        self.steps = steps
        self.params = params
        self.results: Union[None, Dict[str, pd.DataFrame]] = None
        self.category_steps = {}
        for step in self.steps:
            self.category_steps.setdefault(step[0], [])
            self.category_steps[step[0]].append(step[1])

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
        for col in ['p-corr', 'p-unc', 'pval']:
            if col in df.columns:
                return df[df[col] < 0.05]

    def export_statistics(self, file_path: path_t):
        writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
        workbook = writer.book
        header_format = workbook.add_format({'bold': True})
        param_df = pd.DataFrame(self.params.values(), index=self.params.keys(), columns=['parameter'])
        param_df.to_excel(writer, sheet_name="parameter")
        for key, df in self.results.items():
            df.to_excel(writer, sheet_name=key, startrow=1)
            worksheet = writer.sheets[key]
            worksheet.write_string(0, 0, MAP_NAMES[key], header_format)
        writer.save()
