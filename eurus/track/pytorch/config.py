from terrarium import config_value

from eurus.track.config import TrackerConfig


class ForwardTrackerConfig(TrackerConfig):
    r"""
    """
    @property
    @config_value
    def weights(self):
        r"""
        The weights of the model.

        :rtype: str
        """
        return self.RequiredConfig()
