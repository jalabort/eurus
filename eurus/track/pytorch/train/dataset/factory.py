from .alov import AlovPair, AlovSequence, AlovPairConfig, AlovSequenceConfig
from .uav import UavPair, UavSequence, UavPairConfig, UavSequenceConfig
from .vot import VotPair, VotSequence, VotPairConfig, VotSequenceConfig


def create_dataset(config):
    r"""
    Create a tracking dataset from a config.
    
    Parameters
    ----------
    config : :class:`eurus.track.pytorch.train.dataset.TrackingDatasetConfig`
        The configuration for the tracking dataset we're creating.
        
    Returns
    -------
    tracker : :class:`eurus.track.pytorch.train.dataset.SequenceDataset`
        An instance of the tracking dataset created from config.
    """
    if isinstance(config, AlovPairConfig):
        return AlovPair(**config.configuration)
    if isinstance(config, AlovSequenceConfig):
        return AlovSequence(**config.configuration)
    elif isinstance(config, UavPairConfig):
        return UavPair(**config.configuration)
    elif isinstance(config, UavSequenceConfig):
        return UavSequence(**config.configuration)
    elif isinstance(config, VotPairConfig):
        return VotPair(**config.configuration)
    elif isinstance(config, VotSequenceConfig):
        return VotSequence(**config.configuration)
    raise NotImplementedError('No dataset implemented for {}'.format(
        config.__class__))
