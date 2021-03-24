"""

Author: Arthur Wesley

data collection program

"""

import os
import time as t
import random
import sys

import numpy as np
from scipy import stats

from twitchdl import twitch

from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import save_img

from PIL import Image

from src import constants
from src.Data_Collection import web_scrapper
from src.Preprocessing import cropper

temp_images = os.path.join("Data", "Temp Images")


class DataCollector:

    def __init__(self,
                 video_id,
                 step=2,
                 end_transition_step=constants.end_transition_step,
                 verbose=True,
                 dimensions=constants.dimensions,
                 batch_size=32):
        """

        initializes a DataCollector from a specified video ID

        :param video_id: ID of the video to seek through
        :param step: step between images
        :param batch_size: size of the batches that the video is processed in
        """

        # copy the parameters into the object
        self.video_id = video_id
        self.verbose = verbose
        self.batch_size = batch_size
        self.dimensions = dimensions

        # set the steps
        self.step = step
        self.end_transition_step = end_transition_step

        # load NNs
        self.classifier = models.load_model(constants.game_classifier)
        self.crewmate_identifier = models.load_model(constants.crewmate_identifier)

        # get video information & vods
        self.access_token = twitch.get_access_token(video_id)

        self.url = web_scrapper.get_base_url(self.video_id,
                                             self.access_token,
                                             quality=constants.quality(self.dimensions))
        self.full_vods = web_scrapper.get_vods(self.video_id,
                                               self.access_token)

        # take every [step] vods
        self.vods = self.full_vods[::self.step]

        self.vods = [(vod, 0) for vod in self.vods]

        # batches object (initially None)
        self.batches = None

        # transitions object
        self.transitions = None

        # game_classifier_predictions object
        self.game_classifier_predictions = None

        # object containing all of the games in the stream
        self.games = None

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
        for i in range(5):
            try:
                return web_scrapper.get_still_frame(self.url + vod[0],
                                                     vod[1])
            except IndexError:
                print("failed to find the image... try " + str(i + 1) + "/5", file=sys.stderr)

    def get_game_class_batch(self):
        """

        puts all of the images into the vods tensor

        :return:
        """

        t0 = t.time()

        self.batches = [self.vods[i:i + self.batch_size]
                        for i in range(0, len(self.vods), self.batch_size)]

        self.game_classifier_predictions = np.empty((len(self.vods)))

        for index, batch in enumerate(self.batches):

            start_index = self.batch_size * index
            end_index = start_index + len(batch)

            print(start_index)

            # get the tensor
            vods_tensor = self.get_batch(batch)

            # update the game classifier predictions
            self.game_classifier_predictions[start_index:end_index] = np.argmax(self.classifier.predict(vods_tensor), axis=1)

        t1 = t.time()

        if self.verbose:
            print("downloading the images took", t1 - t0)

    def get_batch(self, batch):
        """

        put the specified batch into the vods_tensor

        :param batch: batch to put into the vods tensor
        :return:
        """

        vods_tensor = np.empty((len(batch),) + self.dimensions + (3,))

        for i, vod in enumerate(batch):
            vods_tensor[i] = self.get_image(vod)

        return vods_tensor

    def get_transitions(self):
        """

        generates a list of every transition in the game

        :return: list of each transition in the game
        """

        # get the game classifier predictions if we don't have them
        if self.game_classifier_predictions is None:
            self.get_game_class_batch()

        t0 = t.time()

        self.transitions = list()

        for i in range(len(self.game_classifier_predictions)):

            # if we experienced a transition
            if self.game_classifier_predictions[i] != self.game_classifier_predictions[i - 1]:
                self.transitions.append((constants.label_ids[self.game_classifier_predictions[i]],
                                         i * self.step))

        t1 = t.time()

        if self.verbose:
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

        if self.verbose:
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
                                                       self.end_transition_step,
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

        if self.verbose:
            print("Downloading and classifying the transition images took", t1 - t0, "seconds")

    def save_predictions(self):
        """

        saves the game_classifier_predictions into temp_images

        :return:
        """

        if self.game_classifier_predictions is None:
            self.get_game_class_batch()

        for i, vod in enumerate(self.vods):

            # get the image at the specified index
            image = self.get_image(vod)

            name = constants.label_ids[self.game_classifier_predictions[i]] + "-" + \
                   self.video_id + "-" + \
                   str(i * self.step) + ".jpg"

            save_img(os.path.join(temp_images, name), image)

    def save_transition_predictions(self, over_only=False):
        """

        saves images located at the predicted ends of games

        :param over_only: whether or not to only save over images
        :return: None
        """

        if self.transition_predictions is None:
            self.get_transition_predictions()

        t0 = t.time()

        game_transitions = self.get_game_transitions()

        # number of frames that each transition yields
        frames_per_transition = int(2 * self.step * constants.frames_per_vod / self.end_transition_step)

        # go through the game transitions
        for game_transition_index, (vod_kind, vod_start_index) in enumerate(game_transitions):

            # go through all the vods within the next step
            for i in range(2 * self.step):

                # go through all the frames in each vod
                for frame in range(0, constants.frames_per_vod, self.end_transition_step):

                    vod_index = vod_start_index + i

                    frame_offset = int((i * constants.frames_per_vod + frame)
                                       / self.end_transition_step)

                    index = game_transition_index * frames_per_transition + frame_offset

                    if not over_only or self.transition_predictions[index] == 4:

                        image = self.transition_tensor[index]
                        name = constants.label_ids[self.transition_predictions[index]] + "-" + \
                               self.video_id + "-" + \
                               str(vod_index) + "-" + str(frame) + ".jpg"

                        save_img(os.path.join(temp_images, name), image)

        t1 = t.time()

        if self.verbose:
            print("saving", index, "images took", t1 - t0, "seconds")

    def get_winners(self):
        """

        generates a list containing all of the winning colors in
        a given video

        :return:
        """

        if self.transition_predictions is None:
            self.get_transition_predictions()

        winners = []

        t0 = t.time()

        # print the transition predictions

        end_sets = []

        # get the set of all indices of predictions (games
        for i in range(len(self.transition_predictions)):

            if self.transition_predictions[i] == 4:

                # check to see if this is part of a chain
                if self.transition_predictions[i - 1] != 4:
                    # if this is a new chain, add a new list
                    end_sets.append([i])
                else:
                    # otherwise, expand the old chain
                    end_sets[-1].append(i)

        # choose images from each end set and process them
        for end_set in end_sets:

            # take a random sample
            sample = random.sample(end_set, min(constants.end_screen_samples,
                                                len(end_set)))

            predictions = []

            for item in sample:
                # convert the vod tensor to PIL
                image = Image.fromarray(self.transition_tensor[item])

                # crop the images
                cropped = image.crop(constants.winner_identifier_cropping)

                # crop the image into individual crewmates
                crops = np.array([np.array(item) for item in cropper.crop_crewmates(cropped)])

                predictions.append(np.argmax(self.crewmate_identifier.predict(crops), axis=1))

            # get the predictions by taking the mode along the axis
            winners.append(stats.mode(np.array(predictions), axis=0).mode[0])

        t1 = t.time()

        if self.verbose:
            print("identifying the winners took", t1 - t0, "seconds")

        return winners


def main():
    """

    main method

    :return:
    """

    # problem games: "844688434" (210.ts was not 300 frames long)
    #                "839991024" (1496.ts was not 300 frames long
    #                "795375036" (180.tx was not 300 frames long)
    #                "855465677" (246.ts was not 300 frames long)
    # processes images at 51x real time

    """
    games = [line.strip() for line in open("../../games.txt")]

    for game in games:
        collector = DataCollector(game, step=2)
        collector.save_transition_predictions(over_only=True)
        print("saved images for", game)
        print()
    """

    games = ["957801304", "955126485"]

    #

    for game in games:

        collector = DataCollector(game,
                                  step=20,
                                  end_transition_step=40)

        collector.save_predictions()


if __name__ == "__main__":
    main()
