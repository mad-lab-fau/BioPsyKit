:orphan:

Explicit R-peak outlier detector chains
==================================================

``EcgProcessingPipeline`` has a singular ``outlier_detection_algo`` parameter.
Pass one ``BaseRPeakOutlierDetection`` instance directly, or combine several
detectors explicitly with ``RPeakOutlierDetectionChain``. Normal constructors
are used throughout; there is no conversion factory or implicit list wrapping.

.. code-block:: python

   from biopsykit.signals.ecg import EcgProcessingPipeline
   from biopsykit.signals.ecg.outlier_correction import (
       RPeakOutlierDetectionChain,
       RPeakOutlierDetectionPhysiological,
       RPeakOutlierDetectionQuality,
   )

   pipeline = EcgProcessingPipeline(
       outlier_detection_algo=RPeakOutlierDetectionPhysiological(),
       **other_steps,
   )

   chain = RPeakOutlierDetectionChain(
       detectors=[
           ("physiological", RPeakOutlierDetectionPhysiological()),
           ("quality", RPeakOutlierDetectionQuality()),
       ],
   )
   pipeline = EcgProcessingPipeline(outlier_detection_algo=chain, **other_steps)

Configuration
-------------

The chain declares ``_composite_params = ('detectors',)``. Names are explicit,
unique, nonempty strings without ``__`` and remain stable when steps are
reordered. Its sequence and child objects are stored unchanged in the
constructor, preserving native tpcp cloning and parameter handling.

.. code-block:: python

   # Single detector:
   pipeline.set_params(outlier_detection_algo__hr_thresholds=(40, 180))

   # A named detector inside a chain:
   pipeline.set_params(
       outlier_detection_algo__detectors__physiological__hr_thresholds=(40, 180)
   )

The first example applies to a pipeline configured with a single physiological
detector; the second applies to the explicit chain. Named children can also be
replaced through ``set_params``. Chains can contain other chains, and their
parameters remain recursively accessible. Use these parameter paths in tpcp
parameter searches.

Execution and outputs
---------------------

Detectors execute in list order, each on a separate copy of the original ECG
and R-peaks. Their boolean masks are combined with logical OR. No beats are
removed or corrected between detectors; correction runs once after detection.
This retains the previous multi-detector computation.

Each detector must return a dataframe with one boolean ``is_outlier`` column
without missing values, preserving input beat IDs and row order. Invalid masks
raise an error rather than silently align to another set of beats.

The chain exposes:

* ``points_``: combined ``is_outlier`` mask with the original index;
* ``outlier_masks_``: one boolean column per child name, in execution order;
* ``detectors_``: executed named clones with their individual results.

Configured child objects remain unexecuted. The pipeline exposes its executed
leaf detector or chain as ``outlier_detection_algo_`` and the combined mask as
``outlier_mask_``. ``None`` or an empty chain flags no outliers while retaining
the normal correction stage. An empty R-peak table yields an empty boolean mask.

Migration
---------

Replace the former pipeline parameter ``outlier_detection_algos=[...]`` with
``outlier_detection_algo=RPeakOutlierDetectionChain(detectors=[...])`` and give
every child an explicit name. A one-item list can instead become a single
detector instance. Passing a bare list directly is unsupported.

The default BioPsyKit pipeline uses the same six detectors in the same order,
named ``berntson``, ``physiological``, ``correlation``, ``quality``,
``rr_statistics``, and ``rr_diff_statistics``.
