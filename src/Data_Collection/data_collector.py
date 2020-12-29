"""

Author: Arthur Wesley

data collection program

"""

import os
import sys
from multiprocessing import pool

import numpy as np
from twitchdl import twitch
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import save_img

from src import constants
from src.Data_Collection import web_scrapper

temp_images = os.path.join("Data", "Temp Images")


class DataCollector:

    def __init__(self, video_id, step=2, verbose=True, batch_size=32):
        """

        initializes a DataCollector from a specified video ID

        :param video_id: ID of the video to seek through
        :param step: step between images
        :param batch_size: size of the batches that the video is processed in
        """

        # copy the parameters into the object
        self.video_id = video_id
        self.step = step
        self.verbose = verbose
        self.batch_size = batch_size

        # load NNs
        self.classifier = models.load_model(constants.game_classifier)
        self.winner_identifier = models.load_model(constants.winner_identifier)

        # get video information & vods
        self.access_token = twitch.get_access_token(video_id)

        self.url = web_scrapper.get_base_url(self.video_id, self.access_token)
        self.vods = web_scrapper.get_vods(self.video_id, self.access_token)

        # take every [step] vods
        self.vods = self.vods[::self.step]

        self.vods = [(vod, 0) for vod in self.vods]

        # batches object (initially None)
        self.batches = None

        # predictions object
        self.predictions = None

    def get_image(self, index):
        """

        gets the image at the specified index and puts it into the vods tensor

        :param index: index to insert get the image for
        :return: None
        """

        # get the image and assign it
        return web_scrapper.get_still_frame(self.url + self.vods[index][0],
                                             self.vods[index][1])

    def get_images_batch(self):
        """

        puts all of the images into the vods tensor

        :return:
        """

        self.batches = [self.vods[i:i + self.batch_size]
                   for i in range(0, len(self.vods), self.batch_size)]

        self.predictions = np.empty((len(self.vods)))

        for index, batch in enumerate(self.batches):

            start_index = self.batch_size * index
            end_index = start_index + len(batch)

            # get the tensor
            vods_tensor = self.get_batch(batch, start_index)

            # update the predictions
            self.predictions[start_index:end_index] = np.argmax(self.classifier.predict(vods_tensor), axis=1)

    def get_batch(self, batch, start_index):
        """

        put the specified batch into the vods_tensor

        :param batch: batch to put into the vods tensor
        :param start_index: index of the start of the batch
        :return:
        """

        # with pool.Pool() as p:
        #    vods_tensor = p.map(self.get_image,
        #                        range(start_index, start_index + len(batch)))

        vods_tensor = np.empty((len(batch),) + constants.dimensions + (3,))

        for i in range(len(batch)):
            vods_tensor[i] = self.get_image(start_index + i)

        return vods_tensor

    def get_game_transitions(self):
        """

        generates a list of every transition in the game

        :return: list of each transition in the game
        """

        # get the predictions if we don't have them
        if self.predictions is None:
            self.get_images_batch()

        transitions = list()

        for i in range(len(self.predictions)):

            # if we experianced a transition
            if self.predictions[i] != self.predictions[i - 1]:
                transitions.append((constants.label_ids[self.predictions[i]],
                                    i * self.step))

        return transitions

    def save_predictions(self):
        """

        saves the predictions into temp_images

        :return:
        """

        if self.predictions is None:
            self.get_images_batch()

        for i in range(len(self.vods)):

            # get the image at the specified index
            image = self.get_image(i)

            name = constants.label_ids[self.predictions[i]] + "-" + \
                   self.video_id + "-" + \
                   str(i * self.step) + ".jpg"

            save_img(os.path.join(temp_images, name), image)


def main():
    """

    main method

    :return:
    """

    collector = DataCollector("825004778")
    collector.save_predictions()


if __name__ == "__main__":
    main()
