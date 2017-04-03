from terrarium import config_value
from eurus.track.pytorch.train.dataset.config import DatasetConfig


class Vot2016Config(DatasetConfig):
    r"""
    Configuration for creating a 
    :class:`eurus.track.pytorch.train.dataset.Vot2016`.
    """
    @property
    @config_value
    def root(self):
        r"""
        Path to the dataset.

        :rtype: str
        """
        return '/data1/joan/eurus/data/vot2016'
