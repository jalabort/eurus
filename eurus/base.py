import cv2
import time
import logging

from .utils import Box

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def track_video(video_capture, tracker, initial_box, start_time, end_time):
    r"""
    Track a video segment using one of the implemented trackers.

    Parameters
    ----------
    video_capture : cv2.VideoCapture
        The video capture object containing the video to be tracked.
    tracker : :class:`eurus.track.base.Tracker`
        The tracker.
    initial_box : dict
        The dictionary representation of the bounding box defining the
        target on the first image frame.
    start_time : float
        Timestamp in milliseconds where tracking needs to start.
    end_time : float
        Timestamp in milliseconds where tracking needs to stop.

    Returns
    -------
    tracking_path : list[Box]
        List of bounding boxes containing the path of the target.
    """
    time_start = time.time()

    video_capture.set(cv2.CAP_PROP_POS_MSEC, start_time)
    current_time = video_capture.get(cv2.CAP_PROP_POS_MSEC)

    success, image = video_capture.read()
    if not success:
        raise RuntimeError('Unable to retrieve frame associated to'
                           'timestamp: {}'.format(current_time))

    box = Box.from_dict(initial_box)
    tracker.initialize(image, box)

    tracking_path = []
    n_frames = 0

    while video_capture.grab():
        if current_time >= end_time:
            break
        elif current_time >= start_time:
            success, image = video_capture.retrieve()
            if success is False:
                raise RuntimeError('Unable to retrieve frame associated to '
                                   'timestamp: {}'.format(current_time))
            tracking_box = tracker.track(image, current_time)
            tracking_path.append(tracking_box.to_dict())
        current_time = video_capture.get(cv2.CAP_PROP_POS_MSEC)
        n_frames += 1

    time_end = time.time()
    elapsed_time = time_end - time_start
    logger.info('total time: {0:4.4f}, fps: {0:4.4f}' .format(
        elapsed_time, n_frames / elapsed_time))

    return tracking_path
