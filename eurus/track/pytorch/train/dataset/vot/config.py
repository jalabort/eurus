from terrarium import config_value
from eurus.track.pytorch.train.dataset.config import (
    TrackingDatasetConfig,
    PairTrackingDatasetConfig,
    SequenceTrackingDatasetConfig)


class VotConfigTracking(TrackingDatasetConfig):
    r"""
    Configuration for :class:`eurus.track.pytorch.train.dataset.Vot`.
    """
    @property
    @config_value
    def root(self):
        r"""
        Path to the dataset.

        :rtype: str
        """
        return '/data1/joan/eurus/data/vot2016'


class VotPairConfig(VotConfigTracking, PairTrackingDatasetConfig):
    r"""
    Configuration for :class:`eurus.track.pytorch.train.dataset.VotPair`.
    """


class VotSequenceConfig(VotConfigTracking, SequenceTrackingDatasetConfig):
    r"""
    Configuration for :class:`eurus.track.pytorch.train.dataset.VotSequence`.
    """
