"""

Author: Arthur wesley

"""

import os

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image_dataset_from_directory

from src import constants
from src.Models.Winner_Identifier import initalizer


def numpy_from_filename(filename):
    """

    generate a numpy label for a filename of a labeled image

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


def get_color_label(filename, color):
    """

    get a color label from the specified filename

    :param filename: name of the file to get the color from
    :param color: color to get a label for
    :return: binary label for that file
    """

    # get the colors
    color_string = filename.split("-")[0]
    colors = colors_from_color_string(color_string)

    if color in colors:
        return 1
    else:
        return 0


def get_labels_color(directory, color):
    """

    generates labels for a specific color from the files in the given directory

    :param directory: directory containing the files
    :param color: color to get labels for
    :return: labels of the images in the directory
    """

    files = sorted(os.listdir(os.path.join(directory, "ext")))

    return list(map(lambda file: get_color_label(file, color), files))


def gen_dataset(directory, color="RD"):
    """

    generates a winner identifier dataset from a given directory

    :param directory: directory to generate ethe labels from
    :param color: color to generate a dataset for
    :return: winner identifier dataset
    """

    labels = get_labels(directory)

    return image_dataset_from_directory(directory,
                                        image_size=constants.winner_identifier_dimensions,
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

    training_data_wins = gen_dataset(os.path.join("Data",
                                                  "Winner Identifier",
                                                  "winning games",
                                                  "Training Data"))
    test_data_wins = gen_dataset(os.path.join("Data",
                                              "Winner Identifier",
                                              "winning games",
                                              "Test Data"))

    training_data_losses = gen_dataset(os.path.join("Data",
                                                    "Winner Identifier",
                                                    "winning games",
                                                    "Training Data"))
    test_data_losses = gen_dataset(os.path.join("Data",
                                                "Winner Identifier",
                                                "winning games",
                                                "Test Data"))

    training_data = training_data_wins.concatenate(training_data_losses)
    test_data = test_data_wins.concatenate(test_data_losses)

    # shuffle the data
    training_data.shuffle(1000)
    test_data.shuffle(1000)

    split_data = training_data.take(len(training_data) // 2)

    # run for 200 epochs on training and test data
    train_model(split_data, test_data, epochs=200)
    model = train_model(training_data, test_data, epochs=200)

    model.save(constants.winning_winner_identifier)


if __name__ == "__main__":
    main()
