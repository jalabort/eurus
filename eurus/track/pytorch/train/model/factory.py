from .forward import ForwardTrackingModel, ForwardTrackingModelConfig
from .recurrent import RecurrentTrackingModel, RecurrentTrackingModelConfig


def create_tracking_model(config):
    r"""
    Create a tracking model from a config.

    Parameters
    ----------
    config : :class:`eurus.track.pytorch.train.model.config.TrackingModelConfig`
        The configuration for the tracking model we're creating.

    Returns
    -------
    tracker : :class:`eurus.track.pytorch.train.model.base.TrackingModel`
        An instance of the tracking dataset created from config.
    """
    if isinstance(config, ForwardTrackingModelConfig):
        return ForwardTrackingModel(**config.configuration)
    elif isinstance(config, RecurrentTrackingModelConfig):
        return RecurrentTrackingModel(**config.configuration)
    raise NotImplementedError('No tracking model implemented for {}'.format(
        config.__class__))
