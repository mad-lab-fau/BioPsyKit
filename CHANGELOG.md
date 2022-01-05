# Changelog

## Version 0.4.0 - January 05, 2022
### New Features
- `biopsykit.stats.StatsPipeline`:
  - added new functions `StatsPipeline.results_to_latex_table` and `StatsPipeline.stats_to_latex` 
    to convert statistical analysis results to LaTeX tables and to inline LaTeX code, respectively. 
  - changed argument `stats_type` to `stats_effect_type`.  
    **WARNING: This is a breaking change!** The use of `stats_type` is deprecated and will be removed in `0.5.0`!
- `biopsykit.colors`: Removed `biopsykit.colors` module and replaced it with colors from the new `fau_colors` package.  
  **WARNING: This is a breaking change!**
### Bugfixes
- `biopsykit.signals.imu.static_moment_detection`: end indices of static moments are inclusive instead of exclusive
### Misc
- `biopsykit.utils.array_handling`: if `overlap_percent` > 1 it is converted to a fraction [0.0, 1.0]

## Version 0.3.7 - December 03, 2021
### Misc
- Updated dependencies: now using scikit-learn `>=1.0`

## Version 0.3.6 - December 01, 2021
### New Features
- `biopsykit.io.ecg`: Added functions to write a dictionary of dataframes to a series of csv files (and vice versa).
- `biopsykit.classification.model_selection`: Model selection now also supports to perform randomized-search instead 
  of grid-search for hyperparameter optimization of machine learning pipelines 
### Bugfixes
- `biopsykit.questionnaires.psqi`: Fixed bug in computing Pittsburgh Sleep Quality Index (PSQI)

## Version 0.3.5 – November 05, 2021
### Bugfixes
- further bugfixes (incorrect handling of duplicate index values) for Sleep Analyzer import data using
  `biopsykit.io.sleep_analyzer.load_withings_sleep_analyzer_raw_file()`


## Version 0.3.4 – November 04, 2021
### New Features
- added function `biopsykit.utils.time.extract_time_from_filename()` that can parse time information from a filename
- added function `biopsykit.metadata.gender_counts()`
### Bugfixes
- fixed bug when importing Sleep Analyzer data using
  `biopsykit.io.sleep_analyzer.load_withings_sleep_analyzer_raw_file()`
- `biopsykit.io.load_time_log()`: time log dataframes imported using this function will now have "phase" as column name
### Misc
- updated the minimum version of the `pingouin` dependency to 0.5.0 because it's strongly recommended from the authors 
  of `pingouin` (https://pingouin-stats.org/changelog.html#v0-5-0-october-2021)
- switched from `doit` to `poethepoet` as task runner.

## Version 0.3.2 – October 12, 2021
- final version of BioPsyKit for submission to the [Journal of Open Source Software](https://joss.theoj.org/).
- improved documentation (e.g., included Example Gallery)

## Version 0.3.1 – October 5, 2021
- `biopsykit.utils.file_handling.get_subject_dirs()`: replaced `glob`-style matching of regex strings 
  by `re`-style matching. This allows more flexibility when filtering for directories that match a specific pattern. 
- `biopsykit.questionnaires`:
  - fixed wrong implementation of ASQ (`biopsykit.questionnaires.asq`)
  - added modified version of the ASQ (ASQ_MOD, `biopsykit.questionnaires.asq_mod`) according to (Kramer et al., 2019)
- implemented changes as result of JOSS submission, including:
  - improved handling of example data
  - improved interface for multi-test comparison in `StatsPipeline`
  - improved colormap handling
  - new function to load dataframes in long-format (`biopsykit.io.load_long_format_csv`)

## Version 0.3.0 – September 1, 2021
- initial commit
