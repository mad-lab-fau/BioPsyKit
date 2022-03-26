<img src="./docs/_static/logo/biopsykit_logo.png" height="200">

# BioPsyKit

[![PyPI](https://img.shields.io/pypi/v/biopsykit)](https://pypi.org/project/biopsykit/)
[![status](https://joss.theoj.org/papers/4769dbce3a25db943d7e3a23578becd1/status.svg)](https://joss.theoj.org/papers/4769dbce3a25db943d7e3a23578becd1)
![GitHub](https://img.shields.io/github/license/mad-lab-fau/biopsykit)
[![Documentation Status](https://readthedocs.org/projects/biopsykit/badge/?version=latest)](https://biopsykit.readthedocs.io/en/latest/?badge=latest)
[![Test and Lint](https://github.com/mad-lab-fau/BioPsyKit/actions/workflows/test-and-lint.yml/badge.svg)](https://github.com/mad-lab-fau/BioPsyKit/actions/workflows/test-and-lint.yml)
[![codecov](https://codecov.io/gh/mad-lab-fau/BioPsyKit/branch/main/graph/badge.svg?token=IK0QBHQKCO)](https://codecov.io/gh/mad-lab-fau/BioPsyKit)
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
    * NilsPod binary (`.bin`) files (requires [`NilsPodLib`](https://github.com/mad-lab-fau/NilsPodLib))
    * Other sensor types (_coming soon_)
* Splitting data into single study parts (based on time intervals) that will be analyzed separately
* Performing ECG processing, including:
    * R peak detection (using [`Neurokit`](https://github.com/neuropsychology/NeuroKit))
    * R peak outlier removal and interpolation
    * HRV feature computation
    * ECG-derived respiration (EDR) estimation for respiration rate and respiratory sinus arrhythmia (RSA) 
      (_experimental_)
    * Instantaneous heart rate resampling 
    * Computing aggregated results (e.g., mean and standard error) per study part
* Creating plots for visualizing processing results

#### Quick Example
```python
from biopsykit.signals.ecg import EcgProcessor
from biopsykit.example_data import get_ecg_example

ecg_data, sampling_rate = get_ecg_example()

ep = EcgProcessor(ecg_data, sampling_rate)
ep.ecg_process()

print(ep.ecg_result)
```

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
from biopsykit.example_data import get_hr_subject_data_dict_example
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
hr_subject_data_dict = get_hr_subject_data_dict_example()
# add saliva data collected during the whole TSST procedure
tsst.add_saliva_data(saliva_data, saliva_type="cortisol")
# add heart rate data collected during the "TSST" study part
tsst.add_hr_data(hr_subject_data_dict, study_part="TSST")
# compute heart rate results: normalize ECG data relative to "Preparation" phase; afterwards, use data from the 
# "Talk" and "Math" phases and compute the average heart rate for each subject and study phase, respectively
tsst.compute_hr_results(
    result_id="hr_mean",
    study_part="TSST",
    normalize_to=True,
    select_phases=True,
    mean_per_subject=True,
    params={
        "normalize_to": "Preparation",
        "select_phases": ["Talk", "Math"]
    }
)
```

### Statistical Analysis
`BioPsyKit` implements methods for simplified statistical analysis of biopsychological data by offering an 
object-oriented interface for setting up statistical analysis pipelines, displaying the results, and adding 
statistical significance brackets to plots.

#### Quick Example

```python
import matplotlib.pyplot as plt
from biopsykit.stats import StatsPipeline
from biopsykit.plotting import multi_feature_boxplot
from biopsykit.example_data import get_stats_example

data = get_stats_example()

# configure statistical analysis pipeline which consists of checking for normal distribution and performing paired 
# t-tests (within-variable: time) on each questionnaire subscale separately (grouping data by subscale).
pipeline = StatsPipeline(
    steps=[("prep", "normality"), ("test", "pairwise_ttests")],
    params={"dv": "PANAS", "groupby": "subscale", "subject": "subject", "within": "time"}
)

# apply statistics pipeline on data
pipeline.apply(data)

# plot data and add statistical significance brackets from statistical analysis pipeline
fig, axs = plt.subplots(ncols=3)
features = ["NegativeAffect", "PositiveAffect", "Total"]
# generate statistical significance brackets
box_pairs, pvalues = pipeline.sig_brackets(
    "test", stats_effect_type="within", plot_type="single", x="time", features=features, subplots=True
)
# plot data
multi_feature_boxplot(
    data=data, x="time", y="PANAS", features=features, group="subscale", order=["pre", "post"],
    stats_kwargs={"box_pairs": box_pairs, "pvalues": pvalues}, ax=axs
)
```


### Machine Learning Analysis
`BioPsyKit` implements methods for simplified and systematic evaluation of different machine learning pipelines.

#### Quick Example
```python
# Utils
from sklearn.datasets import load_breast_cancer
# Preprocessing & Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# Classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# Cross-Validation
from sklearn.model_selection import KFold

from biopsykit.classification.model_selection import SklearnPipelinePermuter

# load example dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# specify estimator combinations
model_dict = {
    "scaler": {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler()
    },
    "reduce_dim": {
        "SelectKBest": SelectKBest(),
    },
    "clf" : {
        "KNeighborsClassifier": KNeighborsClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
    }
}
# specify hyperparameter for grid search
params_dict = {
    "StandardScaler": None,
    "MinMaxScaler": None,
    "SelectKBest": { "k": [2, 4, "all"] },
    "KNeighborsClassifier": { "n_neighbors": [2, 4], "weights": ["uniform", "distance"] },
    "DecisionTreeClassifier": {"criterion": ['gini', 'entropy'], "max_depth": [2, 4] },
}

pipeline_permuter = SklearnPipelinePermuter(model_dict, params_dict)
pipeline_permuter.fit(X, y, outer_cv=KFold(5), inner_cv=KFold(5))

# print mean performance scores for each pipeline and parameter combinations, averaged over all outer CV folds
print(pipeline_permuter.mean_pipeline_score_results())
# print overall best-performing pipeline and the performances over all outer CV folds
print(pipeline_permuter.best_pipeline())
# print summary of all relevant metrics for the best pipeline for each evaluated pipeline combination
print(pipeline_permuter.metric_summary())
```


## Installation

``BioPsyKit`` requires Python >=3.7. First, install a compatible version of Python. Then install ``BioPsyKit`` via pip. 

Installation from [PyPi](https://pypi.org/): 
```bash
pip install biopsykit
```

Installation from [PyPi](https://pypi.org/) with extras 
(e.g., `jupyter` to directly install all required dependencies for the use with Jupyter Lab): 
```bash
pip install "biopsykit[jupyter]"
```

Installation from local repository copy:
```bash
git clone https://github.com/mad-lab-fau/BioPsyKit.git
cd BioPsyKit
pip install .
```


### For Developer

If you are a developer and want to contribute to ``BioPsyKit`` you can install an editable version of the package from 
a local copy of the repository.

BioPsyKit uses [poetry](https://python-poetry.org) to manage dependencies and packaging. Once you installed poetry, 
run the following commands to clone the repository, initialize a virtual env and install all development dependencies:

#### Without Extras
```bash
git clone https://github.com/mad-lab-fau/BioPsyKit.git
cd BioPsyKit
poetry install
```

#### With all Extras (e.g., extended functionalities for IPython/Jupyter Notebooks)
```bash
git clone https://github.com/mad-lab-fau/BioPsyKit.git
cd BioPsyKit
poetry install -E mne -E jupyter 
```

To run any of the tools required for the development workflow, use the `poe` commands of the 
[poethepoet](https://github.com/nat-n/poethepoet) task runner:

```bash
$ poe
docs                 Build the html docs using Sphinx.
format               Reformat all files using black.
format_check         Check, but not change, formatting using black.
lint                 Lint all files with Prospector.
test                 Run Pytest with coverage.
update_version       Bump the version in pyproject.toml and biopsykit.__init__ .
```

----
**Note**: In order to build the documentation, you need to additionally install [pandoc](https://pandoc.org/installing.html).

----

See the [Contributing Guidelines](https://biopsykit.readthedocs.io/en/latest/source/contributing/CONTRIBUTING.html) for further information.



## Examples
See the [Examples Gallery](https://biopsykit.readthedocs.io/en/latest/examples/index.html) for example on how to use BioPsyKit.


## Citing BioPsyKit

If you use `BioPsyKit` in your work, please report the version you used in the text. Additionally, please also cite the corresponding paper:

```
Richer et al., (2021). BioPsyKit: A Python package for the analysis of biopsychological data. Journal of Open Source Software, 6(66), 3702, https://doi.org/10.21105/joss.03702
```


If you use a specific algorithm please also to make sure you cite the original paper of the algorithm! We recommend the following citation style:

```
We used the algorithm proposed by Author et al. [paper-citation], implemented by the BioPsykit package [biopsykit-citation].
```
