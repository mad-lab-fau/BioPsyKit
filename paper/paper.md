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

Biopsychology is a field of psychology that analyzes how biological processes interact with behaviour, emotion, 
cognition, and other mental processes. Biopsychology covers, among others, the topics of sensation and perception, 
emotion regulation, movement (and control of such), sleep and biological rhythms, as well as acute and chronic stress.

To assess the interaction between biological and mental processes a variety of different methods are used in the 
field of biopsychology, such as _electrophysiology_, assessed via biosignals like electrocardiogram (ECG), 
electrodermal activity (EDA), electromyogram (EMG), or electroencephalogram (EEG), _neuroendocrine and inflammatory 
biomarker_, assessed via saliva- and blood-based samples, _self-reports_, assessed via psychological questionnaires, 
as well as _sleep, activity and movement_, assessed via inertial measurement units (IMUs).

These methods are used to collect data either during standardized procedures in the laboratory or in the wild. 
The collected data are typically analyzed using statistical methods, or, more recently, using machine learning methods.

In order to combine all these tools necessary for a researcher in the field of biopsychology into 
one single Python package we developed `BioPsyKit`.


# Statement of need
Researchers in biopsychology often combine different assessment modalities during experiments in order to capture the 
interaction between biological and mental processes. One example might be collecting salivary biomarker 
(e.g., cortisol) during an acute stress protocol and investigating the correlation between biomarker and psychometric 
data assessed via self-reports, such as perceived stress, state anxiety, or positive/negative affect. 
However, currently, there exist no Python package that allows to systematically combine, process, and analyze these 
different data modalities out of one hand by using one common API. For that reason `BioPsyKit` enables researchers to 
write cleaner and reproducible analysis code, export analysis results in a standardized format, and create high 
quality figures with for scientific publications. 

# BioPsyKit Structure

The following section describes the structure and the core modules of `BioPsyKit`. 
An overview is also provided in Figure \autoref{fig:overview}.

![Overview of BioPsyKit.\label{fig:overview}](img/biopsykit_overview_figure.pdf)



## Physiological Signal Analysis 
The module `biopsykit.signals` can be used for the analysis of various (electro)physiological signals 
(ECG, EEG, Respiration, Motion, and more). This includes:

- Classes to create processing pipelines for various physiological signals and for extracting relevant parameters 
  from these signals. For physiological signal processing, `BioPsyKit` internally relies on the `neurokit2` 
  Python library [@Makowski2021], but offers further functionalities (e.g., the possibility to apply different outlier 
  removal techniques R peaks extracted from ECG data).  
- Plotting functions specialized for visualizing different physiological signals.

## Sleep Analysis
The module `biopsykit.sleep` can be used for the analysis of motion data collected during sleep. This includes:

- Different algorithms for sleep/wake detection from wrist-worn activity or IMU data, such as the Cole/Kripke
  [@Cole1992] or the Sadeh algorithm [@Sadeh1994].
- Computation of sleep endpoints from detected sleep and wake phases and functions for plotting sleep processing 
  results (e.g., Figure \autoref{fig:sleep_plot}).
- Functions to import and process data from commercially available sleep trackers (e.g., Withings Sleep Analyzer). 

![Example plot for visualizing computed sleep endpoints on IMU data.\label{fig:sleep_plot}](img/img_sleep_imu_plot.pdf){ width=90% }

## Biomarker Analysis
The module `biopsykit.saliva` can be used for the analysis of saliva-based biomarker, such as cortisol and 
alpha-amylase. This also includes the extraction of relevant parameters characterizing salivary biomarker 
(e.g., area under the curve [@Pruessner2003], slope, maximum increase, and more) and specialized plotting functions.


## Self-report Analysis
The module `biopsykit.questionnaires` can be used for the analysis of psychometric self-reports, assessed via 
questionnaires. This includes:

- Functions to convert, clean, and impute tabular data from questionnaire studies.
- Implementation of various established psychological questionnaires, such as Perceived Stress Scale
  (PSS) [@Cohen1983], Primary Appraisal Secondary Appraisal Scale (PASA) [@Gaab2005] and functions to compute scores 
  from questionnaire data.


## Support for Psychological Protocols
The module `biopsykit.protocols` can be used for analyzing data collected during various psychological protocols.
This includes:

- Protocols for the assessment of acute stress in the laboratory, e.g., Trier Social Stress Test (TSST) 
  [@Kirschbaum1993], Montreal Imaging Stress Task (MIST) [@Dedovic2005]. 
- Protocols for the assessment of biological rhythms in the wild (e.g., Cortisol Awakening Response (CAR)).
- Specialized plotting functions for standardized visualization of data collected during these psychological protocols 
  (such as, heart rate data: Figure \autoref{fig:hr_mist}, saliva data: Figure \autoref{fig:saliva_tsst_mist}).

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
The module `biopsykit.stats` and `biopsykit.classification` can be used for simplified evaluation of 
statistical analyses and machine learning pipelines that are frequently used in biopsychological research. 
`biopsykit.stats` provides functions to easily set up statistical analysis pipelines (using `pingouin` [@Vallat2018]) 
and to visualize and export statistical analysis results in a standardized way (see, for example, 
Figure \autoref{fig:stats_boxplot}). 

`biopsykit.classification` provides functions to set up, optimize and evaluate different machine learning pipelines 
for biopsychological problems.

![Example plot for adding statistical analysis results to boxplots.\label{fig:stats_boxplot}](img/img_questionnaire_panas.pdf){ width=60% }



# Availability
The software is available as a pip installable package (`pip install biopsykit`), as well as on GitHub at: 
https://github.com/mad-lab-fau/BioPsyKit.


# Acknowledgements
We acknowledge contributions from Rebecca Lennartz, Daniel Krauß, Victoria Müller, Martin Ullrich, and Janis Zenkner.
Bjoern M. Eskofier gratefully acknowledges the support of the German Research Foundation (DFG) within the framework of 
the Heisenberg professorship programme (grant number ES 434/8-1). Furthermore, this work was partly supported by the 
DFG collaborative research center EmpkinS (CRC 1483).


# References
