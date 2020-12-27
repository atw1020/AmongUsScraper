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
from tensorflow.keras.preprocessing.image import save_img

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

        # initialize the input tensor
        self.vods_tensor = None

        # initalize the predictions
        self.predictions = None

    def get_image(self, index):
        """

        gets the image at the specified index and puts it into the vods tensor

        :param index: index to insert get the image for
        :return: None
        """

        # get the image and assign it
        image = web_scrapper.get_still_frame(self.url + self.vods[index])

        # transpose the image
        image = image.transpose((1, 0, 2))

        self.vods_tensor[index] = image

    def get_images(self):
        """

        saves all of the images in the temporary directory

        :return:
        """

        self.vods_tensor = np.empty((len(self.vods),) + constants.dimensions + (3,))

        for i in range(len(self.vods)):

            self.get_image(i)

            if i % int(float(len(self.vods)) / 100) == 0:
                print(int(float(i) / len(self.vods) * 100), "% complete", sep="")

        print(self.vods_tensor.shape)

    def classify_images(self):
        """

        classify the images in the vods tensor

        :return: None (updates state
        """

        if self.vods_tensor is None:
            self.get_images()

        self.predictions = self.classifier.predict(self.vods_tensor)

        print(self.predictions[0] == self.predictions[1])
        print(self.predictions[0] is self.predictions[1])

        print(self.predictions[0])
        print(self.predictions[1])

        # use the classifier to predict
        self.predictions = np.argmax(self.predictions, axis=1)

        print(self.predictions)

    def save_predictions(self):
        """

        save the predictions

        :return:
        """

        if self.predictions is None:
            self.classify_images()


def main():
    """

    main method

    :return:
    """

    collector = DataCollector("825004778")
    collector.classify_images()


if __name__ == "__main__":
    main()
