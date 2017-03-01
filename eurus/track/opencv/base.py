import cv2
import numpy as np

from eurus.utils import Box
from eurus.track.base import Tracker


class OpenCvTracker(Tracker):
    r"""
    Tracker class based on OpenCv.
    """
    def __init__(self, algorithm='KCF'):
        self.tracker = cv2.MultiTracker(algorithm)

    def initialize(self, image, box):
        r"""
        """
        h, w, _ = image.shape
        image_coords = np.int64(box.to_numpy() * np.array([[w, h] * 2]))
        self.tracker.add(image, [image_coords])

    def track(self, image, current_time):
        r"""
        """
        _, image_coords = self.tracker.update(image)
        h, w, _ = image.shape
        relative_coords = image_coords / np.asarray([[w, h] * 2])
        return Box(*relative_coords[0], current_time)
