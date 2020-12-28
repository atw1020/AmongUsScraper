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

    def __init__(self, video_id, step=2, verbose=True):
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

        self.vods = [(vod, 0) for vod in self.vods]

        # initialize the input tensor
        self.vods_tensor = None

        # initialize the predictions
        self.predictions = None

        # initialize the indices of games
        self.transitions = None

        self.verbose = verbose

    def get_image(self, index):
        """

        gets the image at the specified index and puts it into the vods tensor

        :param index: index to insert get the image for
        :return: None
        """

        # get the image and assign it
        image = web_scrapper.get_still_frame(self.url + self.vods[index][0],
                                             self.vods[index][1])
        self.vods_tensor[index] = image

    def get_images(self):
        """

        saves all of the images in the temporary directory

        :return:
        """

        self.vods_tensor = np.empty((len(self.vods),) + constants.dimensions + (3,))

        # todo: implement with a thread pool

        for i in range(len(self.vods)):

            self.get_image(i)

            if i % int(float(len(self.vods)) / 20) == 0 and self.verbose:
                print(int(float(i) / len(self.vods) * 100), "% complete", sep="")

        print(sys.getsizeof(self.vods_tensor))

    def classify_images(self):
        """

        classify the images in the vods tensor

        :return: None (updates state
        """

        if self.vods_tensor is None:
            self.get_images()

        # use the classifier to predict
        self.predictions = np.argmax(self.classifier.predict(self.vods_tensor), axis=1)

        return self.predictions

    def save_predictions(self):
        """

        save the predictions

        :return:
        """

        if self.predictions is None:
            self.classify_images()

        for i in range(len(self.vods)):

            output_path = constants.label_ids[self.predictions[i]] + "-" \
                          + self.video_id + "-" \
                          + str(i * self.step) + ".jpg"

            save_img(os.path.join(temp_images, output_path), self.vods_tensor[i])

    def get_game_transitions(self):
        """

        gets the indices of the transitions

        :return:
        """

        if self.predictions is None:
            self.classify_images()

        self.transitions = list()

        # go thorough all the predictions
        for i in range(len(self.predictions) - 1):

            # if we hit a lobby and just hit gameplay or meeting,
            if self.predictions[i] != self.predictions[i - 1]:
                self.transitions.append((constants.label_ids[self.predictions[i]], i))

        return self.transitions


def main():
    """

    main method

    :return:
    """

    collector = DataCollector("825004778")
    transitions = collector.get_game_transitions()

    print("transition indices: ", transitions)


if __name__ == "__main__":
    main()
