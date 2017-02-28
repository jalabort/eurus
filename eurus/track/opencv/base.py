import cv2

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
        self.tracker.add(image, [box.to_numpy()])

    def track(self, image, current_time):
        r"""
        """
        _, coordinates = self.tracker.update(image)
        return Box(*coordinates[0], current_time)
