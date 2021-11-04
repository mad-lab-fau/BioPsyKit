# Changelog

## Version 0.3.4 – November 04 2021
### New Features
- added function `biopsykit.utils.time.extract_time_from_filename()` that can parse time information from a filename
- added function `biopsykit.metadata.gender_counts()`
### Bugfixes
- fixed bug when importing Sleep Analyzer data using `biopsykit.io.sleep_analyzer.load_withings_sleep_analyzer_raw_file()`
- `biopsykit.io.load_time_log()`: time log dataframes imported using this function will now have "phase" as column name
### Misc
- updated the minimum version of the `pingouin` dependency to 0.5.0 because it's strongly recommended from the authors of `pingouin` (https://pingouin-stats.org/changelog.html#v0-5-0-october-2021)
- switched from `doit` to `poethepoet` as task runner.

## Version 0.3.2 – October 12 2021
- final version of BioPsyKit for submission to the [Journal of Open Source Software](https://joss.theoj.org/).
- improved documentation (e.g., included Example Gallery)

## Version 0.3.1 – October 5 2021
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

## Version 0.3.0 – September 1 2021
- initial commit
