from terrarium import config_value
from eurus.track.pytorch.train.dataset.config import (
    TrackingDatasetConfig,
    PairTrackingDatasetConfig,
    SequenceTrackingDatasetConfig)


class AlovConfigTracking(TrackingDatasetConfig):
    r"""
    Configuration for :class:`eurus.track.pytorch.train.dataset.Alov`.
    """
    @property
    @config_value
    def root(self):
        r"""
        Path to the dataset.

        :rtype: str
        """
        return '/data1/joan/eurus/data/alov300'


class AlovPairConfig(AlovConfigTracking, PairTrackingDatasetConfig):
    r"""
    Configuration for :class:`eurus.track.pytorch.train.dataset.AlovPair`.
    """


class AlovSequenceConfig(AlovConfigTracking, SequenceTrackingDatasetConfig):
    r"""
    Configuration for :class:`eurus.track.pytorch.train.dataset.AlovSequence`.
    """
