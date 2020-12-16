"""

Author: Arthur Wesley

"""

from twitchdl import twitch

from src.python.Data_Collection import web_scrapper

min_frame_step = 20

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

    def mid_frame(self):
        """

        gets the frame halfway between the upper and lower frame

        :return: frame halfway between upper and lower frames
        """

        return (self.starting_frame + self.ending_frame) / 2

    def mid_index(self):
        """

        gets the index halfway between the current upper and lower index

        :return: vod index halfway between the upper and lower index
        """

        return (self.start_index + self.end_index) / 2

    def search(self, first_half=True):
        """

        update the limits on the vod or frame index and return the image halfway through them

        searches the *second* half of the current range

        :param first_half: whether to search the first or second half of the remaining frames
        :return: frame halfway thorough new range
        """

        # check to see if we are looking at frames or indices

        if self.start_index == self.end_index:

            # check to see if an ending frame currently exists
            if self.ending_frame is None:
                # todo: get the last frame in the sequence
                self.starting_frame = 0
                self.ending_frame = 0
                pass
            elif self.end_index - self.start_index < min_frame_step * 2:
                # stop searching for the end screen
                self.looking_for_end_screen = False

            # update the range
            if first_half:
                self.ending_frame = self.mid_frame()
            else:
                self.starting_frame = self.mid_frame()

            url = self.base_url + self.vods[self.start_index]

            # return the frame
            return web_scrapper.get_still_frame(url, self.mid_frame())

        else:

            # update the range
            if first_half:
                self.end_index = self.mid_index()
            else:
                self.start_index = self.mid_index()

            url = self.base_url + self.vods[self.mid_index()]

            return web_scrapper.get_still_frame(url)

    def next_image(self, image_kind):
        """

        get the next image based on what kind of image this image was

        :param image_kind: kind of image this was
        :return: the next image in the sequence
        """

        if self.looking_for_end_screen:

            # check to see if we reached the over screen
            if image_kind == "Over":
                if self.previous_kind == "Lobby":
                    pass
                if self.previous_kind == "Gameplay" or "Meeting":
                    pass
            elif image_kind == "Lobby":
                # if we
                pass
            elif image_kind == "Gameplay" or "Meeting":
                pass
            else:
                pass

        else:

            self.ending_frame = None

            # first check to see if the we just entered a lobby from gameplay or meeting

            if image_kind == "Lobby" and self.previous_kind == "Gameplay" or "Meeting":

                # if we go from gameplay or a meeting into a lobby it means the game ended
                # and we switch to the looking for end screen state

                self.looking_for_end_screen = True

                # set the indices for the binary search
                self.end_index = self.start_index
                self.start_index = self.start_index - vod_steps["Gameplay"]

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
