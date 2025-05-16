from biopsykit.signals._base_extraction import BaseExtraction

__all__ = ["BaseBPointExtraction", "bpoint_algo_docfiller"]

from biopsykit.utils._docutils import make_filldoc
from biopsykit.utils.dtypes import BPointDataFrame, CPointDataFrame, HeartbeatSegmentationDataFrame, IcgRawDataFrame

bpoint_algo_docfiller = make_filldoc(
    {
        "base_parameters": """
        handle_missing_events : one of {"warn", "raise", "ignore"}, optional
            How to handle failing event extraction. Can be one of:
                * "warn": issue a warning and set the event to NaN
                * "raise": raise an ``EventExtractionError``
                * "ignore": ignore the error and continue with the next event
            Default: "warn"
        """,
        "base_attributes": """
        Attributes
        ----------
        points_ : :class:`~biopsykit.utils.dtypes.BPointDataFrame`
            DataFrame containing the extracted B-points. Each row contains the B-point location
            (in samples from beginning of signal) for each heartbeat, index functions as id of heartbeat.
            B-point locations can be NaN if no B-points were detected for certain heartbeats.
        """,
    }
)


class BaseBPointExtraction(BaseExtraction):
    """Base class for B-point extraction algorithms."""

    points_: BPointDataFrame

    def extract(
        self,
        *,
        icg: IcgRawDataFrame,
        heartbeats: HeartbeatSegmentationDataFrame,
        c_points: CPointDataFrame,
        sampling_rate_hz: float | None,
    ):
        raise NotImplementedError("This is an abstract method and needs to be implemented in a subclass.")
