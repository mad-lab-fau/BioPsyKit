"""Module providing functions to load pretrained models for ICG event extraction algorithms.

Pretrained models are typically too large to ship inside the Python package or the git repository itself.
Instead, they are hosted as assets attached to a GitHub Release of the
`pepbench-experiments <https://github.com/empkins/pepbench-experiments>`_ repository (where they were trained) and
downloaded into the local user folder (``~/.biopsykit_data``) the first time they are requested, using
`pooch <https://www.fatiando.org/pooch/>`_ for the actual fetch-and-cache logic (the same approach already used
for larger example datasets in :mod:`pepbench.example_data`). Pooch verifies each downloaded file against a known
SHA256 hash before handing back its path, so a corrupted or tampered download is caught immediately instead of being
silently loaded.

Models themselves are stored using `skops <https://skops.readthedocs.io/>`_'s persistence format (``.skops``)
rather than :mod:`pickle`. Unlike a raw pickle file, loading a ``.skops`` file does not execute arbitrary code:
the file only ever contains a restricted set of known-safe types (numpy arrays, scikit-learn estimators, and basic
Python containers), and any type outside that safe list must be explicitly whitelisted by the caller before it is
constructed. This matters here because the model is downloaded from the internet and loaded automatically.

Note that neither skops nor the hash verification above protects against *version drift*: a fitted
:class:`~sklearn.ensemble.RandomForestRegressor` embeds internal scikit-learn C-extension objects
(:class:`~sklearn.tree._tree.Tree`) whose binary layout can change between scikit-learn releases. Loading an
artifact trained on a different scikit-learn version than the one currently installed can still fail or silently
misbehave. To make that failure mode legible instead of silent, every artifact records the scikit-learn version
it was trained with, and :func:`get_b_point_abelstuehler2026_model` warns (but does not refuse to load) when the
installed version differs.
"""

import functools
import warnings
from pathlib import Path

import pooch
import sklearn
import skops.io as sio
from packaging.version import InvalidVersion, Version

__all__ = ["get_b_point_abelstuehler2026_model"]

_MODEL_DATA_PATH_HOME = Path.home().joinpath(".biopsykit_data", "pretrained_models")

# GitHub Release tag under which pretrained model assets are attached. Hosted on pepbench-experiments (the repo
# that trained this model, under b_point_ml_experiments/) rather than on BioPsyKit itself, to keep large trained
# artifacts out of the library repo's releases.
# See: https://github.com/empkins/pepbench-experiments/releases/tag/b-point-abelstuehler2026-v1
_B_POINT_ABELSTUEHLER2026_RELEASE_URL = (
    "https://github.com/empkins/pepbench-experiments/releases/download/b-point-abelstuehler2026-v1/"
)

#: Known model artifacts, keyed by ``(feature_set, rater)``, mapping to (file name, SHA256 hash). The hash is
#: verified by pooch on every download, and re-checked (cheaply) against the cached file on every subsequent call.
_B_POINT_ABELSTUEHLER2026_REGISTRY: dict[tuple[str, str], tuple[str, str]] = {
    ("full", "rater_01"): (
        "b_point_abelstuehler2026_full_features_rater_01.skops",
        "sha256:b72246f61245b185963d4e0bb84c2c89331e5532f6b28cd091605b687976f7f3",
    ),
}

_POOCH = pooch.create(
    # Keep using BioPsyKit's existing cache folder convention (~/.biopsykit_data) instead of pooch's own default
    # OS-cache location, so all BioPsyKit-managed downloads live in one place.
    path=_MODEL_DATA_PATH_HOME,
    base_url=_B_POINT_ABELSTUEHLER2026_RELEASE_URL,
    registry=dict(_B_POINT_ABELSTUEHLER2026_REGISTRY.values()),
    # Allows overriding the cache location, e.g. on shared machines or in CI.
    env="BIOPSYKIT_DATA_DIR",
)

#: Types that are expected (and safe) to appear in the AbelStuehler2026 model artifacts, beyond skops' built-in
#: safe list. These are internal scikit-learn/numpy C-extension types used by fitted tree-based estimators
#: (e.g. :class:`~sklearn.ensemble.RandomForestRegressor`); they hold no executable code and are trusted
#: automatically instead of requiring every caller to pass ``trust_model=True`` blindly.
_TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",
    "sklearn.tree._classes.DecisionTreeRegressor",
    "numpy.dtype",
]

#: If the installed scikit-learn's *major.minor* version differs from the one an artifact was trained with by
#: more than this many minor versions, the mismatch warning is escalated from "FYI" to "this may well break".
#: Patch versions (e.g. 1.7.0 vs 1.7.1) never warn - only major/minor differences do, matching scikit-learn's own
#: `InconsistentVersionWarning` granularity.
_VERSION_DRIFT_WARN_THRESHOLD_MINOR = 2


def _check_sklearn_version_compatibility(artifact_sklearn_version: str | None, model_path: str | Path) -> None:
    """Warn if the installed scikit-learn version differs from the one the artifact was trained with."""
    if not artifact_sklearn_version:
        warnings.warn(
            f"The pretrained model at '{model_path}' does not record which scikit-learn version it was trained "
            f"with (installed: scikit-learn {sklearn.__version__}). It was likely created with an older version "
            "of biopsykit. If you encounter errors or unexpected predictions, consider re-downloading it after "
            "clearing the local cache.",
            stacklevel=3,
        )
        return

    installed = Version(sklearn.__version__)
    try:
        trained = Version(artifact_sklearn_version)
    except InvalidVersion:
        warnings.warn(
            f"Could not parse the scikit-learn version recorded in '{model_path}' "
            f"({artifact_sklearn_version!r}); skipping the version-compatibility check.",
            stacklevel=3,
        )
        return

    if installed.release[:2] == trained.release[:2]:
        # same major.minor -> patch differences only, not a known source of breakage
        return

    installed_ord = installed.release[0] * 1000 + installed.release[1]
    trained_ord = trained.release[0] * 1000 + trained.release[1]
    minor_diff = abs(installed_ord - trained_ord)
    if minor_diff > _VERSION_DRIFT_WARN_THRESHOLD_MINOR:
        severity = "may fail or produce incorrect predictions"
    else:
        severity = "may behave slightly differently"
    warnings.warn(
        f"The pretrained model at '{model_path}' was trained with scikit-learn {artifact_sklearn_version}, but "
        f"scikit-learn {sklearn.__version__} is installed. Loading fitted tree-based models (e.g. "
        f"RandomForestRegressor) across scikit-learn versions {severity}, since internal object formats used "
        "by such models are not guaranteed to be stable across releases. If you run into issues, install "
        f"scikit-learn {artifact_sklearn_version} (`pip install scikit-learn=={artifact_sklearn_version}`) or "
        "retrain/re-export the model with your current version.",
        stacklevel=3,
    )


def _load_skops_artifact(model_path: str | Path, trusted: list[str] | None = None) -> dict:
    """Load a skops-serialized artifact, auditing its contents against the trusted-type list first.

    Raises a clear error (instead of silently refusing to load, or silently trusting everything) if the file
    contains a type that is neither part of skops' built-in safe list nor explicitly trusted here.

    """
    trusted = list(_TRUSTED_TYPES) if trusted is None else trusted
    untrusted_types = sio.get_untrusted_types(file=model_path)
    unexpected = [t for t in untrusted_types if t not in trusted]
    if unexpected:
        raise ValueError(
            f"Refusing to load '{model_path}': it contains type(s) {unexpected} that are not in the expected, "
            "trusted set for AbelStuehler2026 model artifacts. This could mean the file is corrupted or was "
            "tampered with. If you are certain the file is trustworthy (e.g. you created it yourself), load it "
            f"manually via `skops.io.load('{model_path}', trusted={untrusted_types})`."
        )
    return sio.load(model_path, trusted=trusted)


@functools.cache
def get_b_point_abelstuehler2026_model(rater: str = "rater_01", feature_set: str = "full") -> object:
    """Load the pretrained B-point regressor for ``BPointExtractionAbelStuehler2026``.

    The returned model is meant to be used with
    :class:`~biopsykit.signals.icg.event_extraction.BPointExtractionAbelStuehler2026`.

    The model is downloaded from the pepbench-experiments GitHub Release on first use (via
    `pooch <https://www.fatiando.org/pooch/>`_, which verifies the download against a known SHA256 hash) and
    cached under ``~/.biopsykit_data/pretrained_models`` for subsequent calls. It is stored using skops'
    persistence format rather than :mod:`pickle`, so loading it does not execute arbitrary code
    (see module docstring).

    If the installed scikit-learn version differs from the one the model was trained with, a :exc:`UserWarning`
    is issued (see module docstring for why this matters); the model is still loaded and returned.

    Parameters
    ----------
    rater : str, optional
        Which rater's labels the model was trained on. Currently, only ``"rater_01"`` is available.
        Default: ``"rater_01"``.
    feature_set : one of {"full", "three_best"}, optional
        Which feature set the model was trained on:

        - ``"full"``: RR-interval + all 12 classical B-point algorithms (13 features).
        - ``"three_best"``: RR-interval + the three most informative classical algorithms (4 features),
          cf. the permutation-importance analysis in the accompanying pepbench experiments.

        Default: ``"full"``.

    Returns
    -------
    object
        A fitted, scikit-learn-compatible regressor (a :class:`~sklearn.pipeline.Pipeline` of a
        :class:`~sklearn.preprocessing.MinMaxScaler` and a :class:`~sklearn.ensemble.RandomForestRegressor`)
        that can be passed directly as the ``model`` parameter of
        :class:`~biopsykit.signals.icg.event_extraction.BPointExtractionAbelStuehler2026`.

    Raises
    ------
    ValueError
        If no pretrained model is registered for the given ``(feature_set, rater)`` combination, or if the
        downloaded/cached model file contains unexpected (untrusted) object types (which could indicate a
        corrupted or tampered file).

    Examples
    --------
    >>> from biopsykit.signals.icg.event_extraction import (
    ...     BPointExtractionAbelStuehler2026,
    ... )
    >>> algo = BPointExtractionAbelStuehler2026(model=get_b_point_abelstuehler2026_model())
    >>> algo.extract(icg=icg, heartbeats=heartbeats, c_points=c_points, sampling_rate_hz=fs)  # doctest: +SKIP

    """
    key = (feature_set, rater)
    if key not in _B_POINT_ABELSTUEHLER2026_REGISTRY:
        available = sorted(_B_POINT_ABELSTUEHLER2026_REGISTRY.keys())
        raise ValueError(
            f"No pretrained model available for feature_set='{feature_set}', rater='{rater}'. "
            f"Available combinations (feature_set, rater): {available}."
        )

    file_name, _file_hash = _B_POINT_ABELSTUEHLER2026_REGISTRY[key]
    model_path = _POOCH.fetch(file_name, progressbar=True)

    artifact = _load_skops_artifact(model_path)

    if isinstance(artifact, dict) and "model" in artifact:
        _check_sklearn_version_compatibility(artifact.get("sklearn_version"), model_path)
        return artifact["model"]
    return artifact
