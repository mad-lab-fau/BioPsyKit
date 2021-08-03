# BioPsyKit

[![PyPI](https://img.shields.io/pypi/v/biopsykit)](https://pypi.org/project/biopsykit/)
![GitHub](https://img.shields.io/github/license/mad-lab-fau/biopsykit)
[![Test and Lint](https://github.com/mad-lab-fau/BioPsyKit/actions/workflows/test-and-lint.yml/badge.svg)](https://github.com/mad-lab-fau/BioPsyKit/actions/workflows/test-and-lint.yml)
![Coverage](./coverage-badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI - Downloads](https://img.shields.io/pypi/dm/biopsykit)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/mad-lab-fau/biopsykit)

A Python package for the analysis of biopsychological data.

With this package you have everything you need for analyzing biopsychological data, including:
* Data processing pipelines for biosignals (ECG, EEG, ...)
* Methods for analyzing saliva samples (cortisol, amylase)
* Implementation of various psychological and HCI-related questionnaires

 
Additionally, there are modules to analyze and visualize data acquired from special measurement scenarios, such as:
* Montreal Imaging Stress Task (MIST)
* ... more to follow

## Details

### Biosignal Analysis
#### ECG Processing
`BioPsyKit` provides a whole ECG data processing pipeline, consisting of:
* Loading ECG data from:
    * generic `.csv` files
    * NilsPod binary (`.bin`) files (requires `NilsPodLib`: https://github.com/mad-lab-fau/NilsPodLib)
    * from other sensor types (_coming soon_)
* Splitting data into chunks (based on time intervals) that will be analyzed separately
* Perform ECG processing, including:
    * R peak detection (using `Neurokit`: https://github.com/neuropsychology/NeuroKit)
    * R peak outlier removal and interpolation
    * HRV feature computation
    * ECG-derived respiration (EDR) estimation for respiration rate and respiratory sinus arrhythmia (RSA) (_experimental_)
* Visualization of results

... more biosignals coming soon!

### Biomarker Analysis
`BioPsyKit` provides several methods for the analysis of biomarkers, such as:
* Load saliva data (e.g. cortisol and amylase) from deepwell plate Excel exports
* Compute standard features (maximum increase, slope, AUC, ...)

### Questionnaires
`BioPsyKit` implements various established psychological and HCI-related questionnaires, such as:
* Perceived Stress Scale (PSS)
* Positive Appraisal Negative Appraisal Scale (PANAS)
* Self-Compassion Scale (SCS)
* System Usability Scale (SUS)
* NASA Task Load Index (NASA-TLX)
* Short Stress State Questionnaire (SSSQ)
* ...

For more details, see the instructions in the `questionnaire` module.

### Stress Protocols
`BioPsyKit` implements methods for analyzing data recorded with several established stress protocols, such as:
* Montreal Imaging Stress Task (MIST)
* Trier Social Stress Test (TSST) (_coming soon..._) 



## Installation
Install it via pip:

```
pip install biopsykit
```


## For developer

```bash
git clone https://github.com/mad-lab-fau/BioPsyKit.git
cd biopsykit
poetry install
```
Install Python >3.8 and [poetry](https://python-poetry.org).
Then run the commands below to get the latest source and install the dependencies:


To run any of the tools required for the development workflow, use the doit commands:

```bash
$ poetry run doit list
docs                 Build the html docs using Sphinx.
format               Reformat all files using black.
format_check         Check, but not change, formatting using black.
lint                 Lint all files with Prospector.
test                 Run Pytest with coverage.
update_version       Bump the version in pyproject.toml and biopsykit.__init__ .
```


## Examples
See Examples in the function documentations on how to use this library.
