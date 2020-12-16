"""

Arthur Wesley

"""

from enum import Enum

from twitchdl import twitch

from src.python.Data_Collection import web_scrapper

#
# constants
#


class ImageGenerator:

    index = 0
    looking_for_end_screen = False

    def __init__(self, video_id):

        access_token = twitch.get_access_token(video_id)

        self.base_url = web_scrapper.get_base_url(video_id, access_token)
        self.vods = web_scrapper.get_vods(video_id, access_token)

    def next_image(self, image_kind):

        pass
