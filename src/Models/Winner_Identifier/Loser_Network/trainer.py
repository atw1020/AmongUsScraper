"""

Author: Arthur wesley

"""

import os

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image_dataset_from_directory

from src import constants
from src.Models.Winner_Identifier.Loser_Network import initalizer


def numpy_from_filename(filename):
    """

    filename of a labeled image

    :param filename: name of the file that you want to get the label from
    :return: numpy array colors the file uses
    """

    # get the colors
    color_string = filename.split("-")[0]
    colors = colors_from_color_string(color_string)

    result = np.zeros(12)

    for color in colors:
        result[constants.color_codes[color]] = 1

    return result


def colors_from_color_string(color_string):
    """

    generates a list of all the colors in a given color string (2 letter color codes
    concatenated together)

    :param color_string: 2 letter color codes concatanated together
    :return: list of two letter color codes
    """

    color_codes = []

    for i in range(0, len(color_string), 2):
        color_codes.append(color_string[i:i+2])

    return color_codes


def get_labels(directory):
    """

    gets the labels for all of the files in the given directory

    :param directory: directory to get the labels for
    :return: labels in order, sorted by name
    """

    # files = sorted(os.listdir(os.path.join(directory, "ext")))
    files = sorted(os.listdir(os.path.join(directory, "ext")))

    return list(map(numpy_from_filename, files))


def gen_dataset(directory):
    """

    generates a winner identifier dataset from a given directory

    :param directory: directory to generate ethe labels from
    :return: winner identifier dataset
    """

    labels = get_labels(directory)

    return image_dataset_from_directory(directory,
                                        image_size=constants.dimensions,
                                        shuffle=True,
                                        labels=labels)


def train_model(dataset, test_data, epochs=50):
    """

    trains a tensorflow model using

    :param dataset: dataset to train on
    :param test_data: data to test on
    :param epochs: number of epochs to train for
    :return: trained model
    """

    # clear the session so that we can train more than one model
    K.clear_session()

    # initialize the model
    model = initalizer.init_nn()

    # fit the model
    model.fit(dataset, epochs=epochs,
              validation_data=test_data)

    return model


def main():
    """

    main method

    :return: None
    """

    """

    color_codes = {v: k for k, v in constants.color_codes.items()}

    training_data = gen_dataset(os.path.join("Data", "Winner Identifier", "Training Data"))

    x, y = next(iter(training_data))

    for i in range(32):

        color_strings = [color_codes[j] if value == 1 else ""
                         for j, value in enumerate(y[i])]

        save_img("".join(color_strings) + str(i) + ".jpg", x[i])

    """

    training_data = gen_dataset(os.path.join("Data", "Winner Identifier", "Training Data"))
    test_data = gen_dataset(os.path.join("Data", "Winner Identifier", "Test Data"))

    split_data = training_data.take(len(training_data) // 2)

    # run for 200 epochs on training and test data
    # train_model(split_data, test_data, epochs=200)
    model = train_model(training_data, test_data, epochs=15)

    model.save(constants.losing_winner_identifier)


if __name__ == "__main__":
    main()
