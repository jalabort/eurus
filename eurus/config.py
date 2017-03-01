from terrarium import Config, config_value

from eurus.track import OpenCvTrackerConfig


class ServerConfig(Config):
    r"""
    """
    @property
    @config_value
    def tracker_config(self):
        r"""
        Configuration for the tracker.

        :rtype: :class:`eurus.track.config.TrackerConfig`
        """
        return OpenCvTrackerConfig()

    @property
    @config_value
    def video_path(self):
        r"""
        Path to the video file.

        :rtype: str
        """
        return self.RequiredConfig()

    @property
    @config_value
    def initial_box(self):
        r"""
        Dictionary representation of the bounding box defining the target
        on the first image frame.

        :rtype: dict
        """
        return self.RequiredConfig()

    @property
    @config_value
    def start_time(self):
        r"""
        Timestamp in milliseconds where tracking needs to start.

        :rtype: float
        """
        return self.RequiredConfig()

    @property
    @config_value
    def end_time(self):
        r"""
        Timestamp in milliseconds where tracking needs to end.

        :rtype: float
        """
        return self.RequiredConfig()
