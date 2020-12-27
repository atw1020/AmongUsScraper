"""

Author: Arthur Wesley

data collection program

"""

import os
from multiprocessing import pool

import cv2
import numpy as np
from twitchdl import twitch
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image_dataset_from_directory

from src import constants
from src.Data_Collection import web_scrapper

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

        # initalize the input tensor
        self.vods_tensor = np.empty((len(self.vods),) + constants.dimensions + (3,))

    def get_predictions(self):
        """

        labels all of the images in the video

        :return:
        """

        # save all of the images into temp
        self.get_images()

        # clear temp when done
        clean_images(temp_images)

    def get_image(self, index):
        """

        gets the image at the specified index and

        :param index:
        :return:
        """

    def get_images(self):
        """

        saves all of the images in the temporary directory

        :return:
        """

        for i in range(len(self.vods)):

            # get the image and assign it
            image = web_scrapper.get_still_frame(self.url + self.vods[i])

            # reshape the tensor for tensorflow
            image = image.transpose((1, 0, 2))
            # cv2.imwrite("test.jpg", image)
            self.vods_tensor[i] = image

            if i % int(float(len(self.vods)) / 100) == 0:
                print(int(float(i) / len(self.vods) * 100), "% complete", sep="")

        print(self.vods_tensor.shape)


def clean_images(path):
    """

    removes all the images from the specified directory

    :return: None
    """

    # clear the directory
    files = os.listdir(path)

    for file in files:
        os.remove(os.path.join(path, file))


def main():
    """

    main method

    :return:
    """

    collector = DataCollector("825004778")
    collector.get_images()


if __name__ == "__main__":
    main()
