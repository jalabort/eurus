from terrarium import config_value
from eurus.track.pytorch.train.dataset.config import (
    TrackingDatasetConfig,
    PairTrackingDatasetConfig,
    SequenceTrackingDatasetConfig)


class UavConfigTracking(TrackingDatasetConfig):
    r"""
    Configuration for :class:`eurus.track.pytorch.train.dataset.Uav`.
    """
    @property
    @config_value
    def root(self):
        r"""
        Path to the dataset.

        :rtype: str
        """
        return '/data1/joan/eurus/data/UAV123'


class UavPairConfig(UavConfigTracking, PairTrackingDatasetConfig):
    r"""
    Configuration for :class:`eurus.track.pytorch.train.dataset.UavPair`.
    """


class UavSequenceConfig(UavConfigTracking, SequenceTrackingDatasetConfig):
    r"""
    Configuration for :class:`eurus.track.pytorch.train.dataset.UavSequence`.
    """
