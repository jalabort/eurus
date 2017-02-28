from .opencv import OpenCvTracker, OpenCvTrackerConfig


def create_tracker(config):
    r"""
    Create the appropriate tracker from a config.

    Parameters
    ----------
    config : :class:`eurus.track.config.TrackerConfig`
        The config specifying the tracker type and its parameters.

    Returns
    -------
    tracker : :class:`eurus.track.base.Tracker`
        Tracker created from the config.
    """
    if isinstance(config, OpenCvTrackerConfig):
        return OpenCvTracker(**config.configuration)
    raise NotImplementedError("Config not implemented for {}".format(
        config.__class__))
