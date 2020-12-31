"""

Author: Arthur Wesley

data collection program

"""

import os

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
        self.full_vods = web_scrapper.get_vods(self.video_id, self.access_token)

        # take every [step] vods
        self.vods = self.full_vods[::self.step]

        self.vods = [(vod, 0) for vod in self.vods]

        # batches object (initially None)
        self.batches = None

        # predictions object
        self.predictions = None

    def get_image(self, vod):
        """

        gets the image at the specified index and puts it into the vods tensor

        :param vod: vod to get
        :return: None
        """

        # get the image and assign it
        return web_scrapper.get_still_frame(self.url + vod[0],
                                             vod[1])

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
            vods_tensor = self.get_batch(batch)

            # update the predictions
            self.predictions[start_index:end_index] = np.argmax(self.classifier.predict(vods_tensor), axis=1)

    def get_batch(self, batch):
        """

        put the specified batch into the vods_tensor

        :param batch: batch to put into the vods tensor
        :return:
        """

        vods_tensor = np.empty((len(batch),) + constants.dimensions + (3,))

        for i, vod in enumerate(batch):
            vods_tensor[i] = self.get_image(vod)

        return vods_tensor

    def get_transitions(self):
        """

        generates a list of every transition in the game

        :return: list of each transition in the game
        """

        # get the predictions if we don't have them
        if self.predictions is None:
            self.get_images_batch()

        transitions = list()

        for i in range(len(self.predictions)):

            # if we experienced a transition
            if self.predictions[i] != self.predictions[i - 1]:
                transitions.append((constants.label_ids[self.predictions[i]],
                                    i * self.step))

        return transitions

    def get_game_transitions(self):
        """

        get the transitions between different kinds of images

        :return: list of transitoins for each game
        """

        # get the number of transitions
        transitions = self.get_transitions()

        game_transitions = []

        for i in range(len(transitions)):

            # if this image is a lobby and the last image was not a lobby or other
            if transitions[i][0] == "Lobby" and transitions[i - 1][0] in ("Gameplay", "Meeting", "Over"):
                game_transitions.append(transitions[i - 1])

        return game_transitions

    def get_transition_images(self, ending_transition):
        """

        generate the images to check for an end screen given an ending transition

        :param ending_transition: tuple (vod file, frame index)
        :return: batch of frames that could contain the transition
        """

        # get the vods to scan
        index = ending_transition[1]
        vods = self.full_vods[index:index + 2 * self.step]

        items = []

        for vod in vods:
            items.append(web_scrapper.get_still_frames(self.url + vod, 50, 300))

        return np.concatenate(items)

    def get_transition_predictions(self):
        """

        get the transition predictions

        :return:
        """

        game_transitions = self.get_game_transitions()
        print(game_transitions)

        tensors = []

        for transition in game_transitions:
            tensors.append(self.get_transition_images(transition))

        tensor = np.concatenate(tensors)
        predictions = np.argmax(self.classifier.predict(tensor), axis=1)

        print(predictions)

    def save_predictions(self):
        """

        saves the predictions into temp_images

        :return:
        """

        if self.predictions is None:
            self.get_images_batch()

        for i, vod in enumerate(self.vods):

            # get the image at the specified index
            image = self.get_image(vod)

            name = constants.label_ids[self.predictions[i]] + "-" + \
                   self.video_id + "-" + \
                   str(i * self.step) + ".jpg"

            save_img(os.path.join(temp_images, name), image)


def main():
    """

    main method

    :return:
    """

    collector = DataCollector("825004778", step=2)
    collector.get_transition_predictions()


if __name__ == "__main__":
    main()
