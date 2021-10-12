# Changelog

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
