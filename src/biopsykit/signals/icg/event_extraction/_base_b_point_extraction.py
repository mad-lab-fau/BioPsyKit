from typing import Optional

from biopsykit.signals._base_extraction import BaseExtraction

__all__ = ["BaseBPointExtraction"]

from biopsykit.utils.dtypes import BPointDataFrame, CPointDataFrame, HeartbeatSegmentationDataFrame, IcgRawDataFrame


class BaseBPointExtraction(BaseExtraction):
    """Base class for B-point extraction algorithms."""

    points_: BPointDataFrame

    def extract(
        self,
        *,
        icg: IcgRawDataFrame,
        heartbeats: HeartbeatSegmentationDataFrame,
        c_points: CPointDataFrame,
        sampling_rate_hz: Optional[float],
    ):
        raise NotImplementedError("This is an abstract method and needs to be implemented in a subclass.")
