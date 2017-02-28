from abc import ABCMeta, abstractmethod


class Tracker(metaclass=ABCMeta):
    r"""
    Abstract base class for defining object tracker classes.
    """
    @abstractmethod
    def initialize(self, image, boxes):
        r"""
        """
        raise NotImplementedError()

    @abstractmethod
    def track(self, **kwargs):
        r"""
        """
        raise NotImplementedError()
