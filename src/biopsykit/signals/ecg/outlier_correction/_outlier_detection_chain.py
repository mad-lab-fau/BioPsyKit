"""Composable R-peak outlier detection with named tpcp algorithms."""

from collections.abc import Sequence

import pandas as pd
from tpcp import Parameter
from typing_extensions import Self

from biopsykit.signals.ecg.outlier_correction._base_outlier_detection import BaseRPeakOutlierDetection

__all__ = ["RPeakOutlierDetectionChain"]


class RPeakOutlierDetectionChain(BaseRPeakOutlierDetection):
    """Run named detectors in order and combine their boolean masks with OR.

    Each detector receives a separate copy of the original ECG and R-peaks. No beats are removed or corrected
    between detectors. Configured algorithms remain unexecuted; clones are retained in ``detectors_``.

    Parameters
    ----------
    detectors : sequence of (str, BaseRPeakOutlierDetection), optional
        Named detectors in execution order. Names must be unique, nonempty, and must not contain ``__``.
        The sequence is stored unchanged for tpcp cloning and nested configuration. An empty sequence flags
        no outliers. Nested chains are supported. Default: ().

    Attributes
    ----------
    points_ : pandas.DataFrame
        Combined boolean ``is_outlier`` column, indexed exactly like the input R-peaks.
    outlier_masks_ : pandas.DataFrame
        Individual boolean masks, one column per configured detector name, in execution order.
    detectors_ : list of (str, BaseRPeakOutlierDetection)
        Executed clones, including their individual results.

    Examples
    --------
    Configure a child with ``chain.set_params(detectors__physiological__hr_thresholds=(40, 180))``.
    Within an ECG pipeline the parameter path is
    ``outlier_detection_algo__detectors__physiological__hr_thresholds``.
    """

    _composite_params = ("detectors",)

    detectors: Parameter[Sequence[tuple[str, BaseRPeakOutlierDetection]]]
    detectors_: list[tuple[str, BaseRPeakOutlierDetection]]
    outlier_masks_: pd.DataFrame

    def __init__(self, detectors: Sequence[tuple[str, BaseRPeakOutlierDetection]] = ()) -> None:
        self.detectors = detectors
        super().__init__()

    def detect_outlier(self, *, ecg: pd.DataFrame, rpeaks: pd.DataFrame, sampling_rate_hz: float) -> Self:
        """Detect outliers without changing configured detectors or input data.

        Masks must contain a boolean ``is_outlier`` column without missing values and preserve the input
        index, including its order. Misaligned masks raise an error instead of being silently aligned.
        """
        for name in ("points_", "outlier_masks_", "detectors_"):
            self.__dict__.pop(name, None)
        self._validate_detectors()
        if not rpeaks.index.is_unique:
            raise ValueError("R-peak indices must be unique.")

        masks = {}
        executed = []
        for name, algorithm in self.detectors:
            detector = algorithm.clone()
            detector.detect_outlier(
                ecg=ecg.copy(deep=True), rpeaks=rpeaks.copy(deep=True), sampling_rate_hz=sampling_rate_hz
            )
            masks[name] = _validate_outlier_mask(detector.points_, rpeaks.index, name)
            executed.append((name, detector))

        self.outlier_masks_ = pd.DataFrame(masks, index=rpeaks.index, dtype=bool)
        self.points_ = self.outlier_masks_.any(axis=1).to_frame("is_outlier")
        self.detectors_ = executed
        return self

    def _validate_detectors(self) -> None:
        if not isinstance(self.detectors, Sequence) or isinstance(self.detectors, (str, bytes)):
            raise TypeError("detectors must be a sequence of (name, detector) pairs.")
        names = set()
        for entry in self.detectors:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise TypeError("Each detector entry must be a (name, detector) pair.")
            name, detector = entry
            if not isinstance(name, str) or not name or "__" in name or name in names:
                raise ValueError("Detector names must be unique, nonempty strings without '__'.")
            if not isinstance(detector, BaseRPeakOutlierDetection):
                raise TypeError(f"Detector '{name}' must inherit BaseRPeakOutlierDetection.")
            names.add(name)


def _validate_outlier_mask(points: pd.DataFrame, index: pd.Index, name: str) -> pd.Series:
    if not isinstance(points, pd.DataFrame) or list(points.columns).count("is_outlier") != 1:
        raise ValueError(f"Detector '{name}' must return a DataFrame with one 'is_outlier' column.")
    if not index.is_unique or not points.index.equals(index):
        raise ValueError(f"Detector '{name}' must preserve the R-peak index and order.")
    mask = points["is_outlier"]
    if not pd.api.types.is_bool_dtype(mask.dtype) or mask.isna().any():
        raise ValueError(f"Detector '{name}' must return a boolean mask without missing values.")
    mask = mask.astype(bool).copy()
    mask.index = index.copy()
    return mask
