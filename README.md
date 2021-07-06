# BioPsyKit

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
pip install git+https://github.com/mad-lab-fau/BioPsyKit.git --upgrade
```

With ssh access:

```
pip install git+ssh://git@github.com:mad-lab-fau/BioPsyKit.git --upgrade
```

For development:

```
git clone https://github.com/mad-lab-fau/BioPsyKit.git
cd BioPsyKit
pip install -r requirements-dev.txt -e . --upgrade
```


## Examples
See Examples in the function documentations on how to use this library.
