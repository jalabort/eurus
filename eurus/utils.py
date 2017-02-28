import numpy as np


class Box(object):
    r"""
    Bounding box class.

    Parameters
    ----------
    x : float
        The horizontal coordinate of the top left vertex of the bounding box
        relative to the image width.
    y : float
        The vertical coordinate of the top left vertex of the bounding box
        relative to the image height.
    w : float
        The horizontal size of the bounding box relative to the image width.
    h : float
        The vertical size of the bounding box relative to the image height.
    timestamp: float
        The video frame timestamp in milliseconds.
    """
    def __init__(self, x, y, w, h, timestamp):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.timestamp = timestamp

    def to_dict(self):
        r"""
        Return the dictionary representation of the bounding box.

        Returns
        -------
        dictionary: dict
            The dictionary representation of the bounding box.
        """
        return self.__dict__

    @classmethod
    def from_dict(cls, dictionary):
        r"""
        Create a bounding box object from its dictionary representation.

        Parameters
        ----------
        dictionary : dict
            The dictionary representation of a bounding box.

        Returns
        --------
        box: :class:`eurus.utils.Box`
            A new bounding box.
        """
        return cls(**dictionary)

    def to_numpy(self):
        r"""
        Return an array containing the x, y, w, h coordinates of the
        bounding box.

        Returns
        -------
        coordinates: np.ndarray
            ``(4,)`` array containing the x, y, w, h coordinates of the
            bounding box.
        """
        return np.asarray([self.x, self.y, self.w, self.h])
