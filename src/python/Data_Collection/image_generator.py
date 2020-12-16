"""

Author: Arthur Wesley

"""

from twitchdl import twitch

from src.python.Data_Collection import web_scrapper

max_frame_sep = 20

vod_steps = {
    "Gameplay": 5,
    "Lobby": 5,
    "Meeting": 5,
    "Over": 5,
    "Other": 20
}


class ImageGenerator:

    start_index = 0
    looking_for_end_screen = False

    starting_frame = 0
    ending_frame = 0

    def __init__(self, video_id):
        """

        initializes the image generator

        :param video_id: video ID to generate images for
        """

        access_token = twitch.get_access_token(video_id)

        self.base_url = web_scrapper.get_base_url(video_id, access_token)
        self.vods = web_scrapper.get_vods(video_id, access_token)

        self.end_index = len(self.vods)

    def next_image(self, image_kind):
        """

        get the next image based on what kind of image this image was

        :param image_kind: kind of image this was
        :return: the next image in the sequence
        """

        if self.looking_for_end_screen:

        else:
            pass
