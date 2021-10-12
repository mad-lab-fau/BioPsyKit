---
title: 'BioPsyKit: A Python package for the analysis of biopsychological data'  
tags:
  - Python
  - Psychology
  - Biological Psychology
  - Questionnaires
  - Electrocardiogram
  - Saliva
  - Sleep
  - Cortisol
authors:
  - name: Robert Richer^[corresponding author]  
    orcid: 0000-0003-0272-5403  
    affiliation: 1
  - name: Arne Küderle  
    orcid: 0000-0002-5686-281X  
    affiliation: 1
  - name: Martin Ullrich  
    orcid: 0000-0001-7348-6097  
    affiliation: 1
  - name: Nicolas Rohleder  
    orcid: 0000-0003-2602-517X    
    affiliation: 2
  - name: Bjoern M. Eskofier  
    orcid: 0000-0002-0417-0336  
    affiliation: 1
affiliations:
  - name: Machine Learning and Data Analytics Lab (MaD Lab), Department Artificial Intelligence in Biomedical Engineering (AIBE), Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)  
    index: 1
  - name: Chair of Health Psychology, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)  
    index: 2
date: 01 September 2021
bibliography: paper.bib
---

# Summary
_Biopsychology_ is a field of psychology that analyzes how biological processes interact with behaviour, emotion, cognition, and other mental processes. 
Biopsychology covers, among others, the topics of sensation and perception, emotion regulation, movement (and control of such), sleep and biological rhythms, as well as acute and chronic stress.

To assess the interaction between biological and mental processes a variety of different modalities are used in the field of biopsychology, such as _electrophysiology_, assessed, for instance, via electrocardiography (ECG), electrodermal activity (EDA), or electroencephalography (EEG), _sleep, activity and movement_, assessed via inertial measurement units (IMUs), _neuroendocrine and inflammatory biomarkers_, assessed by saliva and blood samples, as well as _self-reports_, assessed via psychological questionnaires. 

These different modalities are collected either "in the lab", during standardized laboratory protocols, or "in the wild", during unsupervised protocols in home environments. 
The collected data are typically analyzed using statistical methods, or, more recently, using machine learning methods.


While some software packages exist that allow for the analysis of single data modalities, such as electrophysiological data, or sleep, activity and movement data, no packages are available for the analysis of other modalities, such as neuroendocrine and inflammatory biomarker, and self-reports. 
In order to fill this gap, and, simultaneously, to combine all required tools analyzing biopsychological data from beginning to end into one single Python package, we developed `BioPsyKit`.


# Statement of Need
Researchers in biopsychology often face the challenge of analyzing data from different assessment modalities during experiments in order to capture the complex interaction between biological and mental processes. 

One example might be collecting (electro)physiological (e.g., ECG) or salivary biomarkers (e.g., cortisol) during an acute stress protocol, and investigating the correlation between biomarkers and psychometric data assessed via self-reports, such as perceived stress, state anxiety, or positive/negative affect. 
Another example is the assessment of relationships between sleep and neuroendocrine responses in the morning. To assess the beginning and end of sleep periods, as well as other sleep-related parameters, researchers typically use inertial measurement units (IMUs) or activity trackers. 
These data are then combined with psychometric data from self-reports (e.g., sleep quality, stress coping, etc.) and data from saliva samples to assess the cortisol awakening response (CAR) in the morning.

While some packages already address a subset of these different applications, such as `NeuroKit2` [@Makowski2021] for the analysis of (electro)physiological data, `SleepPy` (Python) [@Christakis2019] or GGIR (R) [@Migueles2019] for sleep analysis from accelerometer data, no software package exists that unites all these different, heterogeneous data modalities under one umbrella. 
Furthermore, and to the best of our knowledge, no software packages exist that allow a standardized analysis of neuroendocrine biomarkers without the requirement to write analysis code from scratch. 
Likewise, no software packages that implement established psychological questionnaires, allowing to compute questionnaire (sub)scales from raw questionnaire items, have been published to date.  

For that reason `BioPsyKit` addresses these limitations and offers all necessary building blocks for the analysis of biopsychological data. 
Our software package allows to systematically combine, process, and analyze data from different modalities using one common API. 
This enables researchers to write cleaner and more reproducible analysis code, to export results in a standardized format, and to create high quality figures for scientific publications. 

# BioPsyKit Structure

The following section describes the structure and the core modules of `BioPsyKit`. An overview is also provided in \autoref{fig:overview}.

![Overview of BioPsyKit.\label{fig:overview}](img/biopsykit_overview_figure.pdf)



## Physiological Signal Analysis 
The module `biopsykit.signals` can be used for the analysis of various (electro)physiological signals 
(ECG, EEG, Respiration, Motion, and more). This includes:

- Classes to create processing pipelines for various physiological signals and for extracting relevant parameters from these signals. 
  For physiological signal processing, `BioPsyKit` internally relies on the `NeuroKit2` Python library [@Makowski2021], but offers further functionalities (e.g., the possibility to apply different R peak outlier removal techniques for R peaks extracted from ECG data).  
- Plotting functions specialized for visualizing different physiological signals.

## Sleep Analysis
The module `biopsykit.sleep` can be used for the analysis of motion data collected during sleep. This includes:

- Different algorithms for sleep/wake detection from wrist-worn activity or IMU data, such as the Cole/Kripke [@Cole1992] or the Sadeh algorithm [@Sadeh1994].
- Computation of sleep endpoints from detected sleep and wake phases and functions for plotting sleep processing results (e.g., \autoref{fig:sleep_plot}).
- Functions to import and process data from commercially available sleep trackers (e.g., Withings Sleep Analyzer). 

![Example plot for visualizing computed sleep endpoints on IMU data.\label{fig:sleep_plot}](img/img_sleep_imu_plot.pdf){ width=90% }

## Biomarker Analysis
The module `biopsykit.saliva` can be used for the analysis of saliva-based biomarkers, such as cortisol and alpha-amylase. 
This also includes the extraction of relevant parameters characterizing salivary biomarkers (e.g., area under the curve [@Pruessner2003], slope, maximum increase, and more) and specialized plotting functions.


## Self-Report Analysis
The module `biopsykit.questionnaires` can be used for the analysis of psychometric self-reports, assessed via 
questionnaires. This includes:

- Functions to convert, clean, and impute tabular data from questionnaire studies.
- Implementation of various established psychological questionnaires, such as Perceived Stress Scale
  (PSS) [@Cohen1983], Primary Appraisal Secondary Appraisal Scale (PASA) [@Gaab2005] and functions to compute scores 
  from questionnaire data.


## Support for Psychological Protocols
The module `biopsykit.protocols` provides an object-oriented interface for psychological protocols. 
On the one hand, it serves as data structure to store and access data collected during this psychological protocol from different modalities. 
On the other hand, the object-oriented interface allows to conveniently compute analysis results from the data added to the particular protocol instance, to export results, and to create plots for data visualization. 
This includes:

- Protocols for the assessment of biological rhythms, especially acute stress, in the laboratory, e.g., Trier Social Stress Test (TSST) [@Kirschbaum1993] or Montreal Imaging Stress Task (MIST) [@Dedovic2005]. 
- Protocols for the assessment of biological rhythms in the wild, e.g., Cortisol Awakening Response (CAR).
- Specialized plotting functions for standardized visualization of data collected during these psychological protocols 
  (such as, heart rate data: \autoref{fig:hr_mist}, saliva data: \autoref{fig:saliva_tsst_mist}).

\begin{figure}[!h]
\includegraphics[width=0.5\textwidth]{img/img_ensemble_plot.pdf}
\includegraphics[width=0.5\textwidth]{img/img_hr_mean_plot.pdf}
\caption{Example plots for visualizing heart rate data collected during the MIST.}
\label{fig:hr_mist}
\end{figure}


\begin{figure}[!h]
\includegraphics[width=0.5\textwidth]{img/img_saliva_plot_tsst_multi.pdf}
\includegraphics[width=0.5\textwidth]{img/img_saliva_plot.pdf}
\caption{Example plots for visualizing biomarker data collected during the TSST (left) and the MIST (right).}
\label{fig:saliva_tsst_mist}
\end{figure}


## Simplified Evaluation
The module `biopsykit.stats` and `biopsykit.classification` can be used for simplified evaluation of statistical analyses and machine learning pipelines that are frequently used in biopsychological research. 
`biopsykit.stats` provides functions to easily set up statistical analysis pipelines (using `pingouin` [@Vallat2018]) and to visualize and export statistical analysis results in a standardized way (see, e.g., \autoref{fig:stats_boxplot}). 

`biopsykit.classification` provides functions to set up, optimize and evaluate different machine learning pipelines 
for biopsychological problems.


\begin{figure}[!h]
\centering
\includegraphics[width=0.6\textwidth]{img/img_questionnaire_panas.pdf}
\caption{Example plot for adding statistical analysis results to boxplots.}
\label{fig:stats_boxplot}
\end{figure}


# Availability
The software is available as a pip installable package (`pip install biopsykit`), as well as on GitHub at: 
https://github.com/mad-lab-fau/BioPsyKit.


# Acknowledgements
We acknowledge contributions from Rebecca Lennartz, Daniel Krauß, Victoria Müller, and Janis Zenkner.
Bjoern M. Eskofier gratefully acknowledges the support of the German Research Foundation (DFG) within the framework of 
the Heisenberg professorship programme (grant number ES 434/8-1). Furthermore, this work was partly supported by the 
DFG collaborative research center EmpkinS (CRC 1483).


# References
