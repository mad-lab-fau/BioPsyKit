# biospykit

A Python package for biopsychological data analysis.

With this package you have everything you need for a whole ECG data analysis pipeline (and more), consisting of:
* Load ECG data from NilsPod binary (`.bin`) files (requires NilsPodLib: https://github.com/mad-lab-fau/NilsPodLib)
* Split data into chunks that will be analyzed separately based on time intervals
* Perform ECG processing, including:
    * R peak detection (using Neurokit: https://github.com/neuropsychology/NeuroKit)
    * R peak outlier removal and interpolation
    * HRV feature computation
    * ECG-based respiration estimation for respiration rate and respiratory sinus arrhythmia (RSA) (_experimental_)
* Visualize processing results

Additionally, there are modules to analyze and visualize data acquired from special measurement scenarios, such as:
* Montreal Imaging Stress Task (MIST)
* ... more to follow

## Installation
Install it via pip:

```
pip install git+https://mad-srv.informatik.uni-erlangen.de/aj78ebud/biopsykit.git --upgrade
```

With ssh access:

```
pip install git+ssh://git@mad-srv.informatik.uni-erlangen.de:aj78ebud/biopsykit.git --upgrade
```

For development:

```
git clone https://mad-srv.informatik.uni-erlangen.de/aj78ebud/biopsykit.git
cd biopsykit
pip install -e . --upgrade
```


## Examples
See Examples in the function documentations on how to use this library.
