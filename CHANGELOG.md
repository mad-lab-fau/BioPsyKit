# Changelog

## Version 0.3.1 – xx
- `biopsykit.utils.file_handling.get_subject_dirs()`: replaced `glob`-style matching of regex strings 
  by `re`-style matching. This allows more flexibility when filtering for directories that match a specific pattern. 
- `biopsykit.questionnaires`:
  - fixed wrong implementation of ASQ (`biopsykit.questionnaires.asq`)
  - added modified version of the ASQ (ASQ_MOD, `biopsykit.questionnaires.asq_mod`) according to (Kramer et al., 2019)

## Version 0.3.0 – September 1 2021
- initial commit
