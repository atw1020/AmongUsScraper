"""

Author: Arthur wesley, Gregory Ghiroli

https://github.com/ihabunek/twitch-dl/blob/master/twitchdl/download.py

"""

import m3u8
import requests
import re

import cv2
import numpy as np
from twitchdl import twitch
from twitchdl import download

from src import constants

from twitchdl.commands import _parse_playlists, _get_playlist_by_name, _get_vod_paths


def get_base_url(video_id,
                 access_token=None,
                 quality=constants.quality(constants.res_360p)):
    """

    generates the URL of a video with a given ID

    :param access_token: access token for the video
    :param video_id: ID of the video
    :param quality: quality of the video
    :return: generated URL
    """

    if access_token is None:
        access_token = twitch.get_access_token(video_id)

    playlists_m3u8 = twitch.get_playlists(video_id, access_token)
    playlists = list(_parse_playlists(playlists_m3u8))
    playlist_url = _get_playlist_by_name(playlists, quality)

    return re.sub("/[^/]+$", "/", playlist_url)


def get_vods(video_id, access_token=None):
    """

    generates a list of all the vods for the video

    :param access_token: access token for the video
    :param video_id: ID of the video
    :return: list of vods the video has
    """

    start = None
    end = None

    if access_token is None:
        access_token = twitch.get_access_token(video_id)

    playlists_m3u8 = twitch.get_playlists(video_id, access_token)
    playlists = list(_parse_playlists(playlists_m3u8))
    playlist_uri = _get_playlist_by_name(playlists,
                                         constants.quality(constants.dimensions))

    response = requests.get(playlist_uri)
    response.raise_for_status()
    playlist = m3u8.loads(response.text)

    return _get_vod_paths(playlist, start, end)


def count_frames(url):
    """

    counts the number of frames in the image at the specified URL

    :param url: URL to the image
    :return: number of frames in that image
    """

    vidObj = cv2.VideoCapture(url)

    success = True
    frames = 0

    while success:
        success, image = vidObj.read()
        frames += 1

    return frames - 1


def get_video(url):
    """

    gets the video from a url

    https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html#ac4107fb146a762454a8a87715d9b7c96

    :param url: url the video is at
    :return: size of the video in bytes
    """

    response = requests.get(url, stream=True, timeout=download.CONNECT_TIMEOUT)

    size = 0

    with open("test.ts", "wb+") as target:

        for chunk in response.iter_content(chunk_size=download.CHUNK_SIZE):
            target.write(chunk)
            size += len(chunk)

    return size


def get_still_frame(url, index=0):
    """

    get a still frame from a video url

    :param index: index of the still frame
    :param url: URL to the video
    :return: image
    """

    vidObj = cv2.VideoCapture(url)

    success = True
    image = None

    while index >= 0 and success:
        index -= 1
        success, image = vidObj.read()

    if not success:
        raise IndexError("index out of bounds " + str(index + 1) + " for getting frame at " + url)

    # convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def get_still_frames(url, step=50, frames=300):
    """

    gets multiple still frames from a video

    :param url: url to the vod
    :param step: number of steps between frames
    :param frames: number of frames in the vod (usually 300)
    :return: tensor of still frames
    """

    vidObj = cv2.VideoCapture(url)

    success = True

    index = 0

    images = []

    frame_set = range(0, frames, step)

    while index < frames and success:

        success, image = vidObj.read()

        if index in frame_set:
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        index += 1

    if not success:
        # try again with fewer frames
        return get_still_frames(url, step=step, frames=frames-step)
        # raise Exception("Could not read all frames from URL: ", url)

    return np.array(images)


def get_training_data(video_id, sampling_rate=constants.sampling_rate):
    """

    generates a set of images from a video ID

    :param video_id: ID of the video
    :param sampling_rate: number of items to skip over
    :return: None
    """

    access_token = twitch.get_access_token(video_id)

    base_url = get_base_url(video_id, access_token)
    vods = get_vods(video_id, access_token)

    for i, vod in enumerate(vods):

        # only take 1 in ten frames
        if i % sampling_rate == 0:
            image = get_still_frame(base_url + vod)
            cv2.imwrite("Data/images/" + video_id + "-" + str(i) + ".jpg", image)


def main():
    """

    main method

    :return:
    """

    # ~ 1 in 20 still frames are wins in shofu's stream

    images = get_still_frames("http://dqrpb9wgowsf5.cloudfront.net/263b198d0bd2ccda59ad_thunderblunder777_40807505038_1607054997/360p30/0.ts")
    print(images.shape)


if __name__ == "__main__":
    main()
