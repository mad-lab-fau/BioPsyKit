:orphan:

ECG pipeline results and lifecycle
==================================

``EcgProcessingPipeline`` keeps each signal-processing stage available separately:

* ``ecg_raw_``: a copy of the input recording.
* ``ecg_clean_``: preprocessing output, before peak extraction and correction.
* ``ecg_processed_``: final ECG, including extractor annotations when supplied and
  the correction stage's heart-rate and outlier results.
* ``rpeaks_raw_``: extracted peaks before outlier correction.
* ``outlier_mask_``: the boolean detection mask aligned with ``rpeaks_raw_``.
* ``rpeaks_``: peaks after general outlier correction.
* ``rpeaks_hrv_``: peaks used for HRV extraction, after optional HRV-specific
  correction. If only HRV correction is enabled, its output is retained here.
  If both HRV steps are disabled, this is ``None``.
* ``hrv_extracted_``: extracted HRV features, or ``None`` when extraction is disabled.

Migration: code that previously read the final annotated ECG from ``ecg_clean_``
must use ``ecg_processed_``. ``ecg_clean_`` now consistently means preprocessing
output. Extractor annotations such as ``r_peak_indicator`` describe the original
detected events; correction does not regenerate those indicators.

Custom peak extractors that only expose ``points_`` remain supported. When no
``ecg_processed_`` is supplied, correction receives a copy of ``ecg_clean_``.

Executed algorithms
-------------------

Every configured algorithm is cloned before execution. Configured parameters stay
unchanged, and executed children are available as:

* ``preprocessing_algo_``
* ``r_peak_algo_``
* ``outlier_detection_algo_``
* ``outlier_correction_algo_``
* ``hrv_r_peak_correction_algo_``
* ``hrv_extraction_algo_``

Disabled children are represented by ``None``. Inspect these executed children for
their intermediate results. Pipeline output DataFrames are separate copies of
child outputs.

The pipeline's ``handle_missing_events`` parameter is forwarded to every cloned
child that exposes that parameter, including nested detector chains. The pipeline
policy takes precedence over a configured child's policy for this execution;
the configured child itself remains unchanged. This forwarding does not change
the individual algorithms' behavior when no events are detected.

Repeated and failed runs
------------------------

Each run first clears the previous result set. All enabled stages must finish
successfully before the pipeline publishes new outputs and executed children.
A failed run therefore exposes neither stale results from a previous recording
nor partial results from the failed execution. Exceptions still propagate.
Subsequent successful runs use fresh child clones and replace all optional
outputs, including setting disabled outputs to ``None``.
