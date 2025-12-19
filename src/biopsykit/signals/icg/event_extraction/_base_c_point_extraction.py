from biopsykit.signals._base_extraction import BaseExtraction

__all__ = ["BaseCPointExtraction"]

from biopsykit.utils.dtypes import CPointDataFrame, HeartbeatSegmentationDataFrame, IcgRawDataFrame


class BaseCPointExtraction(BaseExtraction):
    """Base class for C-point extraction algorithms."""

    points_: CPointDataFrame

    def extract(
        self, *, icg: IcgRawDataFrame, heartbeats: HeartbeatSegmentationDataFrame, sampling_rate_hz: float | None
    ):
        raise NotImplementedError("This is an abstract method and needs to be implemented in a subclass.")
