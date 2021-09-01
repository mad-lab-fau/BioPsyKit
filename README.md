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
* Data processing pipelines for various physiological signals (ECG, EEG, Respiration, Motion, ...).
* Algorithms and data processing pipelines for sleep/wake prediction and computation of sleep endpoints 
  based on activity or IMU data.
* Functions to import and process data from sleep trackers (e.g., Withings Sleep Analyzer)
* Functions for processing and analysis of salivary biomarker data (cortisol, amylase).
* Implementation of various psychological and HCI-related questionnaires.
* Implementation of classes representing different psychological protocols 
  (e.g., TSST, MIST, Cortisol Awakening Response Assessment, etc.)
* Functions for easily setting up statistical analysis pipelines.
* Functions for setting up and evaluating machine learning pipelines.
* Plotting wrappers optimized for displaying biopsychological data.

## Details

### Analysis of Physiological Signals
#### ECG Processing
`BioPsyKit` provides a whole ECG data processing pipeline, consisting of:
* Loading ECG data from:
    * Generic `.csv` files
    * NilsPod binary (`.bin`) files (requires `NilsPodLib`: https://github.com/mad-lab-fau/NilsPodLib)
    * Other sensor types (_coming soon_)
* Splitting data into single parts (based on time intervals) that will be analyzed separately
* Perform ECG processing, including:
    * R peak detection (using `Neurokit`: https://github.com/neuropsychology/NeuroKit)
    * R peak outlier removal and interpolation
    * HRV feature computation
    * ECG-derived respiration (EDR) estimation for respiration rate and respiratory sinus arrhythmia (RSA) 
      (_experimental_)
    * Resample instantaneous heart rate data 
    * Compute aggregated results (e.g., mean and standard error) per part
* Create plots for visualizing processing results

... more biosignals coming soon!

### Sleep/Wake Prediction
`BioPsyKit` allows to process sleep data collected from IMU or activity sensors (e.g., Actigraphs). This includes:
* Detection of wear periods
* Detection of time spent in bed
* Detection of sleep and wake phases
* Computation of sleep endpoints (e.g., sleep and wake onset, net sleep duration wake after sleep onset, etc.)


#### Quick Example
```python
import biopsykit as bp
from biopsykit.example_data import get_sleep_imu_example

imu_data, sampling_rate = get_sleep_imu_example()

sleep_results = bp.sleep.sleep_processing_pipeline.predict_pipeline_acceleration(imu_data, sampling_rate)
sleep_endpoints = sleep_results["sleep_endpoints"]

print(sleep_endpoints)
```

### Salivary Biomarker Analysis
`BioPsyKit` provides several methods for the analysis of salivary biomarkers (e.g. cortisol and amylase), such as:
* Import data from Excel and csv files into a standardized format
* Compute standard features (maximum increase, slope, area-under-the-curve, mean, standard deviation, ...)

#### Quick Example
```python
import biopsykit as bp
from biopsykit.example_data import get_saliva_example

saliva_data = get_saliva_example(sample_times=[-20, 0, 10, 20, 30, 40, 50])

max_inc = bp.saliva.max_increase(saliva_data)
# remove the first saliva sample (t=-20) from computing the AUC
auc = bp.saliva.auc(saliva_data, remove_s0=True)

print(max_inc)
print(auc)
```

### Questionnaires
`BioPsyKit` implements various established psychological (state and trait) questionnaires, such as:
* Perceived Stress Scale (PSS)
* Positive and Negative Affect Schedule (PANAS)
* Self-Compassion Scale (SCS)
* Big Five Inventory (BFI)
* State Trait Depression and Anxiety Questionnaire (STADI)
* Trier Inventory for Chronic Stress (TICS)
* Primary Appraisal Secondary Appraisal Scale (PASA)
* ...

#### Quick Example
```python
import biopsykit as bp
from biopsykit.example_data import get_questionnaire_example

data = get_questionnaire_example()

pss_data = data.filter(like="PSS")
pss_result = bp.questionnaires.pss(pss_data)

print(pss_result)
```

#### List Supported Questionnaires
```python
import biopsykit as bp

print(bp.questionnaires.utils.get_supported_questionnaires())
```

### Psychological Protocols
`BioPsyKit` implements methods for easy handling and analysis of data recorded with several established psychological 
protocols, such as:
* Montreal Imaging Stress Task (MIST)
* Trier Social Stress Test (TSST)
* Cortisol Awakening Response Assessment (CAR)
* ...

#### Quick Example
```python
from biopsykit.protocols import TSST
from biopsykit.example_data import get_saliva_example
from biopsykit.example_data import get_mist_hr_example
# specify TSST structure and the durations of the single phases
structure = {
   "Pre": None,
   "TSST": {
       "Preparation": 300,
       "Talk": 300,
       "Math": 300
   },
   "Post": None
}
tsst = TSST(name="TSST", structure=structure)

saliva_data = get_saliva_example(sample_times=[-20, 0, 10, 20, 30, 40, 50])
hr_data = get_mist_hr_example()
# add saliva data collected during the whole TSST procedure
tsst.add_saliva_data(saliva_data, saliva_type="cortisol")
# add heart rate data collected during the "TSST" study part
tsst.add_hr_data(hr_data, study_part="TSST")
```

## Installation
```bash
pip install biopsykit
```


### For Developer

#### Without Extras
```bash
git clone https://github.com/mad-lab-fau/BioPsyKit.git
cd biopsykit
poetry install
```

#### With all Extras (e.g., extended functionalities for IPython/Jupyter Notebooks)
```bash
git clone https://github.com/mad-lab-fau/BioPsyKit.git
cd biopsykit
poetry install -E mne -E jupyter 
```
Install Python >=3.7 and [poetry](https://python-poetry.org).
Then run the commands below to get the latest source and install the dependencies:


To run any of the tools required for the development workflow, use the `doit` commands:

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
