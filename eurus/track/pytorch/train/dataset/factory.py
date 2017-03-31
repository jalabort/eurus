from .alov import Alov300, Alov300Config
from .uav import Uav123, Uav123Config
from .vot import Vot2016, Vot2016Config


def create_dataset(config):
    r"""
    Create a tracking dataset from a config.
    
    Parameters
    ----------
    config : :class:`eurus.track.pytorch.train.dataset.config.DatasetConfig`
        The configuration for the tracking dataset we're creating.
        
    Returns
    -------
    tracker : :class:`eurus.track.pytorch.train.dataset.TrackingDataset`
        An instance of the tracking dataset created from config.
    """
    if isinstance(config, Alov300Config):
        return Alov300(**config.configuration)
    elif isinstance(config, Uav123Config):
        return Uav123(**config.configuration)
    elif isinstance(config, Vot2016Config):
        return Vot2016(**config.configuration)
    raise NotImplementedError('No dataset implemented for {}'.format(
        config.__class__))
