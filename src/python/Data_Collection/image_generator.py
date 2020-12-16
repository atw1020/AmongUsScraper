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

    previous_kind = "lobby"

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
            pass
        else:

            # first check to see if the we just entered a lobby from gameplay or meeting

            if image_kind == "Lobby" and self.previous_kind == "Gameplay" or "Meeting":

                # if we go from gameplay or a meeting into a lobby it means the game ended
                # and we switch to the looking for end screen state

                self.looking_for_end_screen = True
                return self.next_image(image_kind)
            else:

                # otherwise, update the previous image
                self.previous_kind = image_kind

            step = vod_steps[image_kind]

            if step is None:
                step = 10

            self.start_index += step

            url = self.base_url + self.vods[self.start_index]

            return web_scrapper.get_still_frame(url)
