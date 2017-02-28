import cv2


class VideoCache:

    def __init__(self):
        self._cache = {}

    def add(self, video_path):
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            return False
        self._cache[video_path] = video_capture
        return True

    def get(self, video_path):
        if video_path not in self._cache.keys():
            return False
        return self._cache[video_path]

    def remove(self, video_path):
        if video_path not in self._cache.keys():
            return False
        del self._cache[video_path]
        return True
