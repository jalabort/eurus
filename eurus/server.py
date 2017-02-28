from flask import Flask, request, jsonify
from werkzeug.contrib.fixers import ProxyFix

from eurus.base import track_video
from eurus.track import create_tracker
from eurus.cache import VideoCache
from eurus.config import ServerConfig


app = Flask(__name__)
video_cache = VideoCache()


@app.route("/")
def hello():
    return "Eurus tracker-server is up and running."


@app.route("/track", methods=["POST"])
def track():
    r"""
    """
    config = ServerConfig(**request.json)

    tracker = create_tracker(config.tracker_config)

    video_capture = video_cache.get(config.video_path)
    if not video_capture:
        video_cache.add(config.video_path)
        video_capture = video_cache.get(config.video_path)

    if not video_capture:
        return jsonify({
            "message": "Unable to read video file:, {}.".format(
                config.video_path)
        }), 400

    configuration = config.configuration
    configuration.pop('tracker_config')
    configuration.pop('video_path')

    tracking_path = track_video(video_capture, tracker, **configuration)

    return jsonify(tracking_path)


@app.route("/cache", methods=["POST", "DELETE"])
def cache():
    key = request.form["video_path"]
    if request.method == "POST":
        return add_to_cache(key)
    if request.method == "DELETE":
        return delete_from_cache(key)


def add_to_cache(key):
    if video_cache.get(key):
        return jsonify({
            "message": "Existent key, {}.".format(key)
        }), 400

    if not video_cache.add(key):
        return jsonify({
            "message": "Unable to read video file:, {}.".format(key)
        }), 400

    return jsonify({
        'success': True,
        'cacheKey': key
    })


def delete_from_cache(key):
    if not video_cache.remove(key):
        return jsonify({
            "message": "Invalid key, {}.".format(key)
        }), 400

    return jsonify({
        'success': True,
        'cacheKey': key
    })


app.wsgi_app = ProxyFix(app.wsgi_app)

if __name__ == '__main__':
    app.run()
