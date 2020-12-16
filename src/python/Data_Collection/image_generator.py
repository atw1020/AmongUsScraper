"""

Author: Arthur Wesley

"""

import cv2

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

        # update the range
        if first_half:
            self.end_index = self.mid_index()
        else:
            self.start_index = self.mid_index()

        # check to see if we are looking at frames or indices

        if self.start_index == self.mid_index():

            # check to see if an ending frame currently exists
            if self.ending_frame is None:
                self.starting_frame = 0
                self.ending_frame = web_scrapper.count_frames(self.get_url())
                pass
            elif self.end_index - self.start_index < min_frame_step * 2:
                # stop searching for the end screen
                self.looking_for_end_screen = False

            print("EVERYONE, GET IN HERE!")

            # update the range
            if first_half:
                self.ending_frame = self.mid_frame()
            else:
                self.starting_frame = self.mid_frame()

            url = self.get_url()

            # return the frame
            return web_scrapper.get_still_frame(url, self.mid_frame())

        else:

            url = self.base_url + self.vods[self.mid_index()]

            print("saving image at index", self.mid_index())
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

                    # update the previous kind of image now that we've used it
                    self.previous_kind = image_kind

                    return self.search(False)
                if self.previous_kind == "Gameplay" or "Meeting":

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

                # if we go from gameplay or a meeting into a lobby it means the game ended
                # and we switch to the looking for end screen state

                self.looking_for_end_screen = True

                # set the indices for the binary search
                self.end_index = self.start_index
                self.start_index = self.start_index - vod_steps[self.previous_kind]

                return self.next_image(image_kind)
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

            return web_scrapper.get_still_frame(url)


def test():
    """

    testing method

    :return:
    """

    video_id = "829611887"

    generator = ImageGenerator(video_id)

    image = generator.next_image("Other")

    cv2.imwrite("1.png", image)
    image = generator.next_image("Other")

    cv2.imwrite("2.png", image)
    image = generator.next_image("Other")

    cv2.imwrite("3.png", image)
    image = generator.next_image("Lobby")

    cv2.imwrite("4.png", image)
    image = generator.next_image("Meeting")

    cv2.imwrite("5.png", image)
    image = generator.next_image("Meeting")

    cv2.imwrite("6.png", image)
    image = generator.next_image("Meeting")

    cv2.imwrite("7.png", image)
    image = generator.next_image("Gameplay")

    cv2.imwrite("8.png", image)
    image = generator.next_image("Gameplay")

    cv2.imwrite("9.png", image)
    image = generator.next_image("Meeting")

    cv2.imwrite("10.png", image)
    image = generator.next_image("Meeting")

    cv2.imwrite("11.png", image)
    image = generator.next_image("Gameplay")

    cv2.imwrite("12.png", image)
    image = generator.next_image("Meeting")

    cv2.imwrite("13.png", image)
    image = generator.next_image("Gameplay")

    cv2.imwrite("14.png", image)
    image = generator.next_image("Meeting")

    cv2.imwrite("15.png", image)
    image = generator.next_image("Lobby")

    cv2.imwrite("16.png", image)
    image = generator.next_image("Over")

    cv2.imwrite("17.png", image)


def main():
    """

    main testing method

    :return:
    """

    test()


if __name__ == "__main__":
    main()
