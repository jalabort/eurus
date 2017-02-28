from terrarium import config_value

from eurus.track.config import TrackerConfig


class OpenCvTrackerConfig(TrackerConfig):
    r"""
    """
    @property
    @config_value
    def algorithm(self):
        r"""
        The algorithm to be used for tracking.

        :rtype: str
        """
        return 'KCF'
