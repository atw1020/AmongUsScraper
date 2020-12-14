"""

Author: Arthur wesley, Gregory Ghiroli

https://github.com/ihabunek/twitch-dl/blob/master/twitchdl/download.py

"""

import m3u8
import requests
import re

from src.python import constants

from twitchdl import twitch
from twitchdl.commands import _parse_playlists, _get_playlist_by_name, _get_vod_paths


def get_base_url(video_id):
    """

    generates the URL of a video with a given ID

    :param video_id: ID of the video
    :return: generated URL
    """

    # print_out("<dim>Fetching access token...</dim>")
    access_token = twitch.get_access_token(video_id)

    # print_out("<dim>Fetching playlists...</dim>")
    playlists_m3u8 = twitch.get_playlists(video_id, access_token)
    playlists = list(_parse_playlists(playlists_m3u8))
    playlist_uri = _get_playlist_by_name(playlists, constants.quality)

    return re.sub("/[^/]+$", "/", playlist_uri)


def get_vods(video_id):
    """

    generates a list of all the vods for the video

    :param video_id:
    :return:
    """

    start = None
    end = None

    access_token = twitch.get_access_token(video_id)

    playlists_m3u8 = twitch.get_playlists(video_id, access_token)
    playlists = list(_parse_playlists(playlists_m3u8))
    playlist_uri = _get_playlist_by_name(playlists, constants.quality)

    response = requests.get(playlist_uri)
    response.raise_for_status()
    playlist = m3u8.loads(response.text)

    return _get_vod_paths(playlist, start, end)


def main():
    """

    main method

    :return:
    """

    video_id = "829611887"

    print("url of video", video_id, get_base_url(video_id))
    print("vods", get_vods(video_id))


if __name__ == "__main__":
    main()
