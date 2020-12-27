"""

Author: Arthur Wesley

data collection program

"""

import os

import cv2
from twitchdl import twitch
from tensorflow.keras import models

from src.python import constants
from src.python.Data_Collection import web_scrapper


temp_images = os.path.join("Data", "Temp Images", "ext")


class DataCollector:

    def __init__(self, video_id, step=2):
        """

        initializes a DataCollector from a specified video ID

        :param video_id: ID of the video to seek through
        :param step: step between images
        """

        # copy the parameters into the object
        self.video_id = video_id
        self.step = step

        # load NNs
        self.classifier = models.load_model(constants.game_classifier)
        self.winner_identifier = models.load_model(constants.winner_identifier)

        # get video information & vods
        self.access_token = twitch.get_access_token(video_id)

        self.url = web_scrapper.get_base_url(self.video_id, self.access_token)
        self.vods = web_scrapper.get_vods(self.video_id, self.access_token)

        # take every [step] vods
        self.vods = self.vods[::self.step]

    def get_predictions(self):
        """

        labels all of the images in the video

        :return:
        """

        # save all of the images into temp
        self.save_images()



        # clear temp when done
        clean_images(temp_images)

    def save_images(self):
        """

        saves all of the images in the temporary directory

        :return:
        """

        for i in range(len(self.vods)):
            image = web_scrapper.get_still_frame(self.video_id + self.vods[i])
            frame_id = self.video_id + "-" + str(i) + ".jpg"

            cv2.imwrite(os.path.join(temp_images, frame_id), image)


def clean_images(path):
    """

    removes all the images from the specified directory

    :return: None
    """

    # clear the directory
    files = os.listdir(path)

    for file in files:
        os.remove(os.path.join(path, file))

