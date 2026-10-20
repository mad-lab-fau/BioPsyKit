:orphan:

Sleep epoch coverage and endpoints
========================================

``compute_sleep_endpoints()`` accepts variable positive epoch durations in
seconds through its keyword-only ``epoch_length`` argument. Returned durations
and latencies are always in minutes. Classifier outputs carry the epoch
duration in ``predictions.attrs['epoch_length_s']``; both processing pipelines
propagate it to endpoint computation.

Each row represents an epoch starting at its index value and ending one epoch
later. Datetime indices contain epoch-start timestamps. Integer indices count
epochs rather than minutes: four 30-second epochs indexed 0 through 3 cover
``[0, 4)`` and last two minutes. Index names have no effect.

Epoch length is resolved from the explicit argument, prediction metadata, or
regular datetime spacing. These must agree. For a single timestamp without
known frequency or integer indices, supply an explicit duration or metadata.
Gaps, duplicate/reversed timestamps, nonbinary/missing predictions, and bed
intervals outside observed coverage raise errors instead of treating unobserved
time as sleep or wake.

.. code-block:: python

   epoch_length = 30
   bed_interval = [
       predictions.index[0],
       predictions.index[-1] + pd.Timedelta(seconds=epoch_length),
   ]
   endpoints = compute_sleep_endpoints(
       predictions, bed_interval, epoch_length=epoch_length
   )

Bed intervals and sleep/wake bouts have inclusive starts and exclusive ends.
In particular, the end of the final epoch is one epoch after its timestamp.
Legacy callers passing the final included timestamp as the bed end must move
that boundary to the intended exclusive end. If a bed boundary falls inside an
epoch, only its overlapping duration contributes.

Endpoint meanings
-----------------

* ``total_sleep_duration`` follows its documented meaning: the span from the
  first sleep start to the final sleep end, including intervening wake. It no
  longer counts every epoch in bed.
* ``net_sleep_duration`` sums covered sleep time.
* ``time_in_bed`` measures the bed interval. ``sleep_efficiency`` is
  ``100 * net_sleep_duration / time_in_bed``.
* ``wake_onset`` is the exclusive end of the final sleeping epoch. A recording
  entirely classified as sleep therefore has zero sleep-onset and get-up
  latency, rather than one epoch each.
* ``wake_after_sleep_onset`` and ``wake_bouts`` include only wake between first
  and final sleep. Leading and trailing wake contribute to latencies instead.

For four sleeping 30-second epochs with a bed interval covering all four,
``net_sleep_duration`` and ``total_sleep_duration`` are both 2 minutes, and the
sleep bout ends at the start timestamp plus 2 minutes. No observed sleep
retains the empty-dictionary result; ``endpoints_as_df({})`` returns ``None``.

Classification and pipelines
----------------------------

Configure ``SleepWakeDetection(epoch_length=30, ...)`` consistently with the
input activity epochs. The Actigraph pipeline resolves input epoch duration
before constructing the classifier and does not resample precomputed activity.
Direct classifier prediction overrides do not mutate constructor settings.

The acceleration pipeline sums one-second activity counts into complete epochs
of the configured duration. It discards incomplete trailing epochs and padded
counts, and clips its detected bed interval to observed complete-epoch coverage.
Raw aggregation therefore requires whole-second epoch lengths.

Variable endpoint durations do not add new calibrated classifier kernels.
Cole/Kripke supports its existing 10-, 30-, and 60-second kernels. Webster
rescoring supports 30 and 60 seconds; use ``rescore_data=False`` for other epoch
durations, such as Cole/Kripke's 10-second epochs. Unsupported rescoring lengths
now raise a clear error instead of silently taking the 60-second branch.
Sazonov and Scripps Clinic retain the 30-second defaults previously used by
their prediction code; the other algorithms default to 60 seconds.
Constructor epoch settings now reach Sadeh and Sazonov prediction and rescoring.
``ColeKripkeOld`` also uses its configured epoch length for rescoring, replacing
the previous implicit 30-second rescoring of predictions configured for 60 seconds.
