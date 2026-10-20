Sample interval migration
=========================

IMU static-moment intervals and ECG heartbeat intervals use zero-based sample
positions with an inclusive start and an exclusive end: ``[start, end)``.
Positions are relative to the original input signal, independent of its index
labels. A nonempty interval satisfies ``0 <= start < end <= len(data)``.
The end may equal ``len(data)``; it is a boundary, not an indexable sample.

.. code-block:: python

   segment = data.iloc[start:end]
   n_samples = end - start

Adjacent intervals ``[0, 10)`` and ``[10, 20)`` share no samples. A single
sample at position 5 is ``[5, 6)``. Empty detector results contain zero rows
with integer start/end columns; ``[0, 0]`` is not a detected interval.
Overlapping intervals remain separate intervals, including fixed-length
heartbeats whose windows overlap.

Changed outputs
---------------

* ``find_static_moments()`` now returns exclusive ends. A fully static
  100-sample recording produces ``start=0, end=100`` instead of ``end=99``.
* ``bool_array_to_start_end_array()``, ``_find_static_sequences()``, and
  ``find_first_static_window_multi_sensor()`` retain their exclusive ends.
* Fixed-length ``HeartbeatSegmentationNeurokit`` outputs now set
  ``end_sample`` to NeuroKit's last included sample plus one. Incomplete
  segments with padding outside the recording are discarded, matching the
  variable-length mode. Variable-length ends retain their meaning, and a
  complete final heartbeat ending exactly at ``len(ecg)`` is now retained.
* Q-peak assignment includes the start of a heartbeat and excludes its end.
  ICG consumers already use exclusive positional slices and now receive the
  same convention from both heartbeat segmentation modes. PEP output passes
  through the exclusive ``heartbeat_end_sample`` boundary.

Migrating saved intervals
-------------------------

Convert a saved table only when its source used inclusive ends:

.. code-block:: python

   # Legacy output from find_static_moments():
   moments = legacy_moments.copy()
   moments["end"] += 1

   # Legacy fixed-length HeartbeatSegmentationNeurokit output:
   heartbeats = legacy_fixed_heartbeats.copy()
   heartbeats["end_sample"] += 1
   heartbeats = heartbeats.loc[
       (heartbeats["start_sample"] >= 0)
       & (heartbeats["end_sample"] <= len(ecg))
   ]

Do not increment ends from the boolean converter, internal static-sequence
detector, multi-sensor window detector, or variable-length heartbeat segmenter.
Do not convert newly generated outputs. Interval values cannot identify their
convention, so retain the source and version of stored tables. There is no
automatic convention detection.

Remove compensating ``end + 1`` operations from consumers of newly generated
intervals. To access the final included sample, use ``data.iloc[end - 1]``.
Do not access ``data.iloc[end]`` to obtain a duration or terminal timestamp.

Static-moment features
----------------------

``static_moment_duration()`` and ``compute_features()`` now require the
keyword ``sampling_rate_hz``. Durations measure sample coverage:

.. code-block:: python

   duration_s = static_moment_duration(
       data, np.array([start, end]), sampling_rate_hz=100
   )
   features = compute_features(data, moments, sampling_rate_hz=100)
   # duration_s == (end - start) / 100

At 100 Hz, 100 samples cover 1.00 s, replacing the previous 0.99 s difference
between first and last included timestamps. A single sample covers 0.01 s.
Exactly 6,000 samples qualify for the at-least-60-second subset; prior duration
calculations could exclude them. Corrected means include the last detected
sample and may therefore change orientation results.

These functions support regularly sampled recordings. Datetime indices must
agree with the supplied sampling rate, allowing nanosecond rounding. Split
recordings at gaps or resample irregular data before computing these features;
no duration across gaps is inferred from timestamps.

``compute_features()`` requires a datetime index for nonempty data. Its optional
time window is also start-inclusive and end-exclusive. Intervals are clipped to
that window. By default, the window ends one sample period after the final
timestamp. Empty inputs or intervals return zero counts and NaN statistics;
the same rule applies when no interval reaches 60 s.

``mean_orientation()`` returns one row per interval and preserves a dataframe's
interval index, including distinct intervals of equal length. Array inputs use
an ``interval_id`` index. The old index based on interval length is removed.
