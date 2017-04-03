from terrarium import Config, config_value


class DatasetConfig(Config):
    r"""
    Configuration for creating a 
    :class:`eurus.track.pytorch.train.dataset.TrackingDataset`.
    """
    @property
    @config_value
    def root(self):
        r"""
        Path to the dataset.
        
        :rtype: str
        """
        return self.RequiredConfig()

    @property
    @config_value
    def sequence_length(self):
        r"""
        The length of the dataset sequences. If `None` or larger than the 
        length of the shortest sequence in the dataset this configuration 
        value will be equal to the length of the shortest sequence in the 
        dataset. 
        
        :rtype: int | None
        """
        return None

    @property
    @config_value
    def skip(self):
        r"""
        The number of consecutive frames to skip when creating a sequence. 
        If `None` no frames are skipped.
        
        :rtype: int | None
        """
        return None

    @property
    @config_value
    def context_factor(self):
        r"""
        
        :rtype: int
        """
        return 3

    @property
    @config_value
    def search_factor(self):
        r"""
        
        :rtype: int | None
        """
        return 2

    @property
    @config_value
    def context_size(self):
        r"""
        
        :rtype: int | (int, int)
        """
        return 128

    @property
    @config_value
    def search_size(self):
        r"""
        
        :rtype: int | (int, int)
        """
        return 256

    @property
    @config_value
    def response_size(self):
        r"""

        :rtype: int | (int, int)
        """
        return 33

