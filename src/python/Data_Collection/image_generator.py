"""

Author: Arthur Wesley

"""

from twitchdl import twitch
import cv2

from src.python.Data_Collection import web_scrapper

min_frame_step = 20

vod_steps = {
    "Gameplay": 5,
    "Lobby": 5,
    "Meeting": 5,
    "Over": 1,
    "Other": 40
}


class ImageGenerator:

    looking_for_end_screen = False

    starting_frame = 0
    ending_frame = 0

    previous_kind = "lobby"

    def __init__(self, video_id, starting_index=0):
        """

        initializes the image generator

        :param video_id: video ID to generate images for
        """

        self.video_id = video_id
        self.start_index = starting_index
        access_token = twitch.get_access_token(video_id)

        self.base_url = web_scrapper.get_base_url(video_id, access_token)
        self.vods = web_scrapper.get_vods(video_id, access_token)

        self.end_index = len(self.vods)

        self.image = None  # initally None

    def get_url(self):
        """

        gets the URL of the current (first) vod

        :return: URL to the current video
        """

        return self.base_url + self.vods[self.start_index]

    def mid_frame(self):
        """

        gets the frame halfway between the upper and lower frame

        :return: frame halfway between upper and lower frames
        """

        return int((self.starting_frame + self.ending_frame) / 2)

    def mid_index(self):
        """

        gets the index halfway between the current upper and lower index

        :return: vod index halfway between the upper and lower index
        """

        return int((self.start_index + self.end_index) / 2)

    def search(self, first_half):
        """

        update the limits on the vod or frame index and return the image halfway through them

        searches the *second* half of the current range

        :param first_half: whether to search the first or second half of the remaining frames
        :return: frame halfway thorough new range
        """

        # check to see if we are looking at frames or indices

        # todo: fix bug with this binary searcher
        if self.start_index == self.mid_index():

            # check to see if an ending frame currently exists
            if self.ending_frame is None:
                self.starting_frame = 0
                self.ending_frame = web_scrapper.count_frames(self.get_url())
            else:

                # update the range

                if first_half:
                    self.ending_frame = self.mid_frame()
                else:
                    self.starting_frame = self.mid_frame()

                if self.ending_frame - self.starting_frame < min_frame_step * 2:
                    # stop searching for the end screen
                    self.looking_for_end_screen = False

            url = self.get_url()

            print("saving image at index", self.mid_index(), "frame", self.mid_frame())

            self.image = web_scrapper.get_still_frame(url, self.mid_frame())
            # return the frame
            return self.image

        else:

            # update the range
            if first_half:
                self.end_index = self.mid_index()
            else:
                self.start_index = self.mid_index()

            print("start", self.start_index)
            print("end", self.end_index)
            print("middle", self.mid_index())

            url = self.base_url + self.vods[self.mid_index()]

            print("saving image at index", self.mid_index())

            self.image = web_scrapper.get_still_frame(url)

            return self.image

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

                    # update the previous kind of image now that we've used it
                    self.previous_kind = image_kind

                    return self.search(False)
                if self.previous_kind == "Gameplay" or "Meeting":

                    # update the previous kind of image now that we've used it
                    self.previous_kind = image_kind

                    return self.search(True)
                else:

                    # update the previous kind of image now that we've used it
                    self.previous_kind = image_kind

                    return self.search(True)

            elif image_kind == "Lobby":

                # update the previous kind of image now that we've used it
                self.previous_kind = image_kind

                return self.search(True)
            elif image_kind == "Gameplay" or "Meeting":

                # update the previous kind of image now that we've used it
                self.previous_kind = image_kind

                return self.search(False)
            else:

                # this is a strange case that may result from quickly switching
                # between streams. return something in the first half
                return self.search(True)

        else:

            self.ending_frame = None

            # first check to see if the we just entered a lobby from gameplay or meeting

            if image_kind == "Lobby" and (self.previous_kind == "Gameplay"
                                          or self.previous_kind == "Meeting"):

                url = self.get_url()

                # if we go from gameplay or a meeting into a lobby it means the game ended
                # and we switch to the looking for end screen state

                self.looking_for_end_screen = True

                # set the indices for the binary search
                self.end_index = self.start_index
                self.start_index = self.start_index - vod_steps[self.previous_kind]

                self.image = web_scrapper.get_still_frame(url)

                return self.image

            else:

                # otherwise, update the previous image
                self.previous_kind = image_kind

            # update the index
            step = vod_steps[image_kind]

            if step is None:
                step = 10

            self.start_index += step

            # get the next url
            url = self.get_url()
            print("saving image at index", self.start_index)

            self.image = web_scrapper.get_still_frame(url)

            return self.image

    def save_next(self, image_kind):
        """

        save the next image of the specified kind

        :param image_kind: kind of image to save
        :return: None (saves image)
        """

        output = image_kind + "-" + self.video_id + "-"

        if not self.start_index == self.mid_index() and self.looking_for_end_screen:
            output += str(self.mid_index())
        else:
            output += str(self.start_index)

        if self.ending_frame is None:
            output += ".jpg"
        else:
            output += "-" + str(self.mid_frame()) + ".jpg"

        cv2.imwrite(output, self.image)

        self.next_image(image_kind)
