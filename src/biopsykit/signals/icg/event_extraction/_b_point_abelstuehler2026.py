import warnings

import numpy as np
import pandas as pd

from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS, CanHandleMissingEventsMixin
from biopsykit.signals.icg.event_extraction._b_point_arbol2017 import (
    BPointExtractionArbol2017IsoelectricCrossings,
    BPointExtractionArbol2017SecondDerivative,
    BPointExtractionArbol2017ThirdDerivative,
)
from biopsykit.signals.icg.event_extraction._b_point_debski1993 import BPointExtractionDebski1993SecondDerivative
from biopsykit.signals.icg.event_extraction._b_point_drost2022 import BPointExtractionDrost2022
from biopsykit.signals.icg.event_extraction._b_point_forouzanfar2018 import BPointExtractionForouzanfar2018
from biopsykit.signals.icg.event_extraction._b_point_lozano2007 import (
    BPointExtractionLozano2007LinearRegression,
    BPointExtractionLozano2007QuadraticRegression,
)
from biopsykit.signals.icg.event_extraction._b_point_miljkovic2022 import BPointExtractionMiljkovic2022
from biopsykit.signals.icg.event_extraction._b_point_pale2021 import BPointExtractionPale2021
from biopsykit.signals.icg.event_extraction._b_point_sherwood1990 import BPointExtractionSherwood1990
from biopsykit.signals.icg.event_extraction._b_point_stern1985 import BPointExtractionStern1985
from biopsykit.signals.icg.event_extraction._base_b_point_extraction import BaseBPointExtraction, bpoint_algo_docfiller
from biopsykit.signals.icg.event_extraction._pretrained_models import get_b_point_abelstuehler2026_model
from biopsykit.utils.array_handling import sanitize_input_dataframe_1d
from biopsykit.utils.dtypes import (
    BPointDataFrame,
    CPointDataFrame,
    HeartbeatSegmentationDataFrame,
    IcgRawDataFrame,
    is_b_point_dataframe,
    is_c_point_dataframe,
    is_heartbeat_segmentation_dataframe,
    is_icg_raw_dataframe,
)
from biopsykit.utils.exceptions import EventExtractionError

__all__ = ["BPointExtractionAbelStuehler2026"]


@bpoint_algo_docfiller
class BPointExtractionAbelStuehler2026(BaseBPointExtraction, CanHandleMissingEventsMixin):
    """B-point extraction algorithm based on a machine-learning meta-regressor over classical algorithms.

    This algorithm frames B-point detection as a *feature-based regression* problem instead of designing a new
    signal-processing rule: For every heartbeat, the B-point sample estimates of the full set of established,
    classical B-point extraction algorithms available in :mod:`biopsykit` are computed and combined with the
    (estimated) RR-interval into a feature vector. This feature vector is passed to a pretrained,
    scikit-learn-compatible regressor which directly predicts the (heartbeat-relative) location of the B-point in
    milliseconds.

    The feature set consists of the RR-interval and the B-point estimates of the following classical algorithms:

    - :class:`~biopsykit.signals.icg.event_extraction.BPointExtractionArbol2017IsoelectricCrossings`,
    - :class:`~biopsykit.signals.icg.event_extraction.BPointExtractionArbol2017SecondDerivative`,
    - :class:`~biopsykit.signals.icg.event_extraction.BPointExtractionArbol2017ThirdDerivative`,
    - :class:`~biopsykit.signals.icg.event_extraction.BPointExtractionDebski1993SecondDerivative`,
    - :class:`~biopsykit.signals.icg.event_extraction.BPointExtractionDrost2022`,
    - :class:`~biopsykit.signals.icg.event_extraction.BPointExtractionForouzanfar2018`,
    - :class:`~biopsykit.signals.icg.event_extraction.BPointExtractionLozano2007LinearRegression`,
    - :class:`~biopsykit.signals.icg.event_extraction.BPointExtractionLozano2007QuadraticRegression`,
    - :class:`~biopsykit.signals.icg.event_extraction.BPointExtractionMiljkovic2022`,
    - :class:`~biopsykit.signals.icg.event_extraction.BPointExtractionPale2021`,
    - :class:`~biopsykit.signals.icg.event_extraction.BPointExtractionSherwood1990`, and
    - :class:`~biopsykit.signals.icg.event_extraction.BPointExtractionStern1985`.

    This class does **not** train a model. By default (``model=None``), the pretrained model from the original
    experiments is used: it is downloaded on the first call to :meth:`extract` (and cached locally afterwards) via
    :func:`~biopsykit.signals.icg.event_extraction.get_b_point_abelstuehler2026_model`.
    Alternatively, any already-fitted, scikit-learn-compatible regressor that implements ``predict`` (e.g., a
    :class:`~sklearn.pipeline.Pipeline` of a :class:`~sklearn.preprocessing.MinMaxScaler` and a
    :class:`~sklearn.ensemble.RandomForestRegressor`, as used in the original experiments) can be passed via the
    ``model`` parameter. The classmethod
    :meth:`extract_training_features` can be used to build a matching feature matrix and target vector from
    labeled B-point annotations in order to train such a model, e.g. via
    ``RandomForestRegressor().fit(x, y)``.

    Because the underlying regressor was trained on data that contains missing values (whenever one of the base
    algorithms failed to detect a B-point/C-point for a given heartbeat), missing feature values are passed to the
    model as ``NaN`` rather than being imputed or dropped. The ``model`` therefore needs to be able to handle
    missing values natively (e.g., recent versions of :class:`~sklearn.ensemble.RandomForestRegressor` support
    this).

    Parameters
    ----------
    %(base_parameters)s

    %(base_attributes)s

    """

    model: object | None

    #: Base algorithms used to compute the per-heartbeat candidate features (in addition to the RR-interval).
    #: Maps the feature name (as used during model training) to the algorithm class used to compute it.
    _BASE_ALGORITHM_CLASSES: dict[str, type[BaseBPointExtraction]] = {
        "arbol2017-isoelectric-crossings": BPointExtractionArbol2017IsoelectricCrossings,
        "arbol2017-second-derivative": BPointExtractionArbol2017SecondDerivative,
        "arbol2017-third-derivative": BPointExtractionArbol2017ThirdDerivative,
        "debski1993-second-derivative": BPointExtractionDebski1993SecondDerivative,
        "drost2022": BPointExtractionDrost2022,
        "forouzanfar2018": BPointExtractionForouzanfar2018,
        "lozano2007-linear-regression": BPointExtractionLozano2007LinearRegression,
        "lozano2007-quadratic-regression": BPointExtractionLozano2007QuadraticRegression,
        "miljkovic2022": BPointExtractionMiljkovic2022,
        "pale2021": BPointExtractionPale2021,
        "sherwood1990": BPointExtractionSherwood1990,
        "stern1985": BPointExtractionStern1985,
    }

    #: Names (and order) of the features that are passed to :attr:`model`, matching the columns of the training
    #: data used to fit the original model.
    FEATURE_NAMES = ("rr_interval_ms", *_BASE_ALGORITHM_CLASSES.keys())

    def __init__(
        self,
        model: object | None = None,
        handle_missing_events: HANDLE_MISSING_EVENTS = "warn",
    ):
        """Initialize new ``BPointExtractionAbelStuehler2026`` instance.

        See class docstring for parameter details.

        """
        self.model = model
        super().__init__(handle_missing_events=handle_missing_events)

    def extract(
        self,
        *,
        icg: IcgRawDataFrame,
        heartbeats: HeartbeatSegmentationDataFrame,
        c_points: CPointDataFrame,
        sampling_rate_hz: float,
    ):
        """Extract B-points from given ICG derivative signal using the pretrained meta-regressor.

        Parameters
        ----------
        icg : IcgRawDataFrame
            The raw ICG signal data.
        heartbeats : HeartbeatSegmentationDataFrame
            The heartbeat segmentation data.
        c_points : CPointDataFrame
            The C-point data.
        sampling_rate_hz : float
            The sampling rate of the ICG signal in Hz.

        Returns
        -------
        BPointDataFrame
            The extracted B-point data.

        """
        self._check_valid_missing_handling()
        model = self._get_model()
        is_icg_raw_dataframe(icg)
        is_heartbeat_segmentation_dataframe(heartbeats)
        is_c_point_dataframe(c_points)
        icg = sanitize_input_dataframe_1d(icg, column="icg_der")

        start_samples = heartbeats["start_sample"].astype(float)

        features = self._build_feature_matrix(
            icg=icg, heartbeats=heartbeats, c_points=c_points, sampling_rate_hz=sampling_rate_hz
        )

        b_points = pd.DataFrame(index=heartbeats.index, columns=["b_point_sample", "nan_reason"])

        # Heartbeats without a valid start sample cannot be anchored back to an absolute sample position, even
        # though the regressor itself could still produce a (relative) prediction for them.
        missing_start = start_samples.isna()
        predictable_idx = heartbeats.index[~missing_start]

        if len(predictable_idx) > 0:
            x_pred = features.loc[predictable_idx, list(self.FEATURE_NAMES)].to_numpy(dtype=float)
            predicted_ms = np.asarray(model.predict(x_pred)).ravel()
            predicted_samples = start_samples.loc[predictable_idx].to_numpy() + (predicted_ms / 1000 * sampling_rate_hz)
            b_points.loc[predictable_idx, "b_point_sample"] = np.round(predicted_samples)

        b_points.loc[missing_start, "b_point_sample"] = np.nan
        b_points.loc[missing_start, "nan_reason"] = "heartbeat_start_nan"

        idx_nan = b_points["b_point_sample"].isna()
        if idx_nan.sum() > 0:
            idx_nan = list(b_points.index[idx_nan])

            missing_str = (
                f"The heartbeat start sample contains NaN at heartbeats {idx_nan}! The B-points were also set to "
                f"NaN."
            )
            if self.handle_missing_events == "warn":
                warnings.warn(missing_str)
            elif self.handle_missing_events == "raise":
                raise EventExtractionError(missing_str)

        b_points = b_points.astype({"b_point_sample": "Int64", "nan_reason": "object"})
        is_b_point_dataframe(b_points)

        self.points_ = b_points
        return self

    def _get_model(self) -> object:
        # Loaded lazily (instead of as a default argument) so that importing this module never triggers a download
        # and so that instances don't share a mutable default (which tpcp rejects on clone()).
        model = get_b_point_abelstuehler2026_model() if self.model is None else self.model
        if not hasattr(model, "predict"):
            raise AttributeError(
                "The provided 'model' must be a fitted, scikit-learn-compatible regressor that implements "
                "'predict' (e.g., a fitted `sklearn.pipeline.Pipeline` combining a scaler and a regressor)."
            )
        return model

    @classmethod
    def _build_feature_matrix(
        cls,
        *,
        icg: pd.Series,
        heartbeats: HeartbeatSegmentationDataFrame,
        c_points: CPointDataFrame,
        sampling_rate_hz: float,
    ) -> pd.DataFrame:
        """Build the per-heartbeat feature matrix (RR-interval + base-algorithm B-point estimates, all in ms).

        All B-point sample estimates are normalized relative to the start of their heartbeat and converted from
        samples to milliseconds, matching the convention used to build the original training data.

        """
        start_samples = heartbeats["start_sample"].astype(float)

        feature_dict = {"rr_interval_ms": cls._compute_rr_interval_ms(heartbeats, sampling_rate_hz)}

        for feature_name, algo_cls in cls._BASE_ALGORITHM_CLASSES.items():
            algo = algo_cls(handle_missing_events="ignore")
            algo.extract(icg=icg, heartbeats=heartbeats, c_points=c_points, sampling_rate_hz=sampling_rate_hz)
            b_point_sample = algo.points_["b_point_sample"].astype(float)
            feature_dict[feature_name] = (b_point_sample - start_samples) / sampling_rate_hz * 1000

        return pd.DataFrame(feature_dict, index=heartbeats.index)

    @staticmethod
    def _compute_rr_interval_ms(heartbeats: HeartbeatSegmentationDataFrame, sampling_rate_hz: float) -> pd.Series:
        """Compute the RR-interval (to the next heartbeat) in milliseconds.

        Reuses an existing ``rr_interval_ms``/``rr_interval_sample`` column of ``heartbeats`` if present (e.g., as
        computed by :class:`~biopsykit.signals.ecg.segmentation.HeartbeatSegmentationNeurokit`), and otherwise
        computes it from consecutive R-peak locations.

        """
        if "rr_interval_ms" in heartbeats.columns:
            return heartbeats["rr_interval_ms"].astype(float)
        if "rr_interval_sample" in heartbeats.columns:
            return heartbeats["rr_interval_sample"].astype(float) / sampling_rate_hz * 1000

        r_peaks = heartbeats["r_peak_sample"].astype(float)
        rr_interval_sample = r_peaks.diff(periods=-1).abs()
        return rr_interval_sample / sampling_rate_hz * 1000

    @classmethod
    def extract_training_features(
        cls,
        *,
        icg: IcgRawDataFrame,
        heartbeats: HeartbeatSegmentationDataFrame,
        c_points: CPointDataFrame,
        b_points: BPointDataFrame,
        sampling_rate_hz: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a training feature matrix and target vector from labeled B-point annotations.

        For each heartbeat, the same RR-interval and base-algorithm features used by :meth:`extract` are computed,
        and the target is the annotated B-point location, normalized relative to the heartbeat start and converted
        to milliseconds. Heartbeats with a missing reference B-point or a missing heartbeat start sample are
        excluded; heartbeats with (some) missing feature values (e.g., because a base algorithm could not detect a
        C-point) are kept, with the corresponding feature(s) set to ``NaN``, matching the original training
        methodology. The resulting feature matrix and target vector can be used to fit a scikit-learn-compatible
        regressor (e.g., via ``RandomForestRegressor().fit(x, y)``) to be passed as ``model`` to
        :class:`BPointExtractionAbelStuehler2026`.

        Parameters
        ----------
        icg : IcgRawDataFrame
            The raw ICG signal data.
        heartbeats : HeartbeatSegmentationDataFrame
            The heartbeat segmentation data.
        c_points : CPointDataFrame
            The C-point data.
        b_points : BPointDataFrame
            The reference (ground truth) B-point data used as training targets.
        sampling_rate_hz : float
            The sampling rate of the ICG signal in Hz.

        Returns
        -------
        tuple[:class:`~numpy.ndarray`, :class:`~numpy.ndarray`]
            Feature matrix ``x`` of shape ``(n_heartbeats, len(FEATURE_NAMES))``, with columns ordered as in
            :attr:`FEATURE_NAMES`, and target vector ``y`` of shape ``(n_heartbeats,)`` containing the
            heartbeat-relative B-point location in milliseconds.

        """
        is_icg_raw_dataframe(icg)
        is_heartbeat_segmentation_dataframe(heartbeats)
        is_c_point_dataframe(c_points)
        is_b_point_dataframe(b_points)
        icg = sanitize_input_dataframe_1d(icg, column="icg_der")

        start_samples = heartbeats["start_sample"].astype(float)
        reference_b_point = b_points["b_point_sample"].astype(float)

        features = cls._build_feature_matrix(
            icg=icg, heartbeats=heartbeats, c_points=c_points, sampling_rate_hz=sampling_rate_hz
        )
        target_ms = (reference_b_point - start_samples) / sampling_rate_hz * 1000

        valid = target_ms.notna() & start_samples.notna()

        x_train = features.loc[valid, list(cls.FEATURE_NAMES)].to_numpy(dtype=float)
        y_train = target_ms.loc[valid].to_numpy(dtype=float)

        return x_train, y_train
