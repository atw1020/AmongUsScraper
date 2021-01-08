"""

Author: Arthur Wesley

data collection program

"""

import os
import time as t

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

        # transitions object
        self.transitions = None

        # predictions object
        self.predictions = None

        # transition tensor
        self.transition_tensor = None

        # transition predictions
        self.transition_predictions = None

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

        t0 = t.time()

        self.batches = [self.vods[i:i + self.batch_size]
                        for i in range(0, len(self.vods), self.batch_size)]

        self.predictions = np.empty((len(self.vods)))

        for index, batch in enumerate(self.batches):

            start_index = self.batch_size * index
            end_index = start_index + len(batch)
            
            print(start_index)

            # get the tensor
            vods_tensor = self.get_batch(batch)

            # update the predictions
            self.predictions[start_index:end_index] = np.argmax(self.classifier.predict(vods_tensor), axis=1)

        t1 = t.time()
        print("downloading the images took", t1 - t0)

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

        t0 = t.time()

        self.transitions = list()

        for i in range(len(self.predictions)):

            # if we experienced a transition
            if self.predictions[i] != self.predictions[i - 1]:
                self.transitions.append((constants.label_ids[self.predictions[i]],
                                         i * self.step))

        t1 = t.time()
        print("finding the transitions took", t1 - t0)

    def get_game_transitions(self):
        """

        get the transitions between different kinds of images

        :return: list of transitions for each game
        """

        # get the number of transitions
        if self.transitions is None:
            self.get_transitions()

        t0 = t.time()

        game_transitions = []

        for i in range(len(self.transitions)):

            # if this image is a lobby and the last image was not a lobby or other
            if self.transitions[i][0] == "Lobby" and self.transitions[i - 1][0] in ("Gameplay", "Meeting", "Over"):
                game_transitions.append(self.transitions[i - 1])

        t1 = t.time()
        print("finding the game transitions took", t1 - t0, "seconds")

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
            items.append(web_scrapper.get_still_frames(self.url + vod,
                                                       constants.end_transition_step,
                                                       constants.frames_per_vod))

        return np.concatenate(items)

    def get_transition_predictions(self):
        """

        get the transition predictions

        :return:
        """

        game_transitions = self.get_game_transitions()

        t0 = t.time()

        tensors = []

        for transition in game_transitions:
            tensors.append(self.get_transition_images(transition))

        self.transition_tensor = np.concatenate(tensors)
        self.transition_predictions = np.argmax(self.classifier.predict(self.transition_tensor), axis=1)

        t1 = t.time()
        print("classifying the transition images took", t1 - t0, "seconds")

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

    def save_transition_predictions(self, over_only=False):
        """

        saves predictions made about the transition images into temp_images

        :param over_only: whether or not to only save over images
        :return: None
        """

        if self.transition_predictions is None:
            self.get_transition_predictions()

        t0 = t.time()

        game_transitions = self.get_game_transitions()

        # number of frames that each transition yields
        frames_per_transition = int(2 * self.step * constants.frames_per_vod / constants.end_transition_step)

        # go through the game transitions
        for game_transition_index, (vod_kind, vod_start_index) in enumerate(game_transitions):

            # go through all the vods within the next step
            for i in range(2 * self.step):

                # go through all the frames in each vod
                for frame in range(0, constants.frames_per_vod, constants.end_transition_step):

                    vod_index = vod_start_index + i

                    frame_offset = int((i * constants.frames_per_vod + frame)
                                       / constants.end_transition_step)

                    index = game_transition_index * frames_per_transition + frame_offset

                    if not over_only or self.transition_predictions[index] == 4:

                        image = self.transition_tensor[index]
                        name = constants.label_ids[self.transition_predictions[index]] + "-" + \
                               self.video_id + "-" + \
                               str(vod_index) + "-" + str(frame) + ".jpg"

                        save_img(os.path.join(temp_images, name), image)

        t1 = t.time()
        print("saving", index, "images took", t1 - t0, "seconds")


def main():
    """

    main method

    :return:
    """

    # problem games: "844688434" (210.ts was not 300 frames long)
    #                "839991024" (1496.ts was not 300 frames long

    games = ["854705673", "854705960", "795375036",
             "856248469", "854264128", "851948386",
             "850543910", "849582421", "848563869",
             "847450846", "846449518", "845189559",
             "844881191"]

    for game in games:
        collector = DataCollector(game, step=2)
        collector.save_transition_predictions(over_only=True)
        print("saved images for", game)
        print()


if __name__ == "__main__":
    main()
