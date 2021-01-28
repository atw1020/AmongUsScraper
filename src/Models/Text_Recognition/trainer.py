"""

Author: Arthur Wesley

"""

import os

from tensorflow.keras.preprocessing import image_dataset_from_directory

from src.Models.Text_Recognition import text_utils
from src import constants


def get_vocab(directory):
    """

    get the vocab from a directory

    :param directory: directory to get the vocabulary from
    :return: vocabulary from the directory
    """

    files = os.listdir(os.path.join(directory,
                                    "ext"))

    # get the names of all the players
    names = [file.split("-")[2] for file in files]

    return text_utils.get_vocab(names)


def get_labels(directory):
    """

    gets the string labels from the specified

    :param directory: directory to get the labels from
    :return: labels of the images in that directory
    """

    files = os.listdir(os.path.join(directory,
                                    "ext"))

    # get the names of all the players
    names = [file.split("-")[2] for file in files]

    vocab = text_utils.get_vocab(names)

    return [text_utils.label_from_string(name, vocab) for name in names]


def gen_dataset(directory):
    """

    generate a dataset from the

    :param directory:
    :return:
    """

    labels = get_labels(directory)

    return image_dataset_from_directory(directory,
                                        labels=labels,
                                        image_size=constants.meeting_dimensions)


def train_model(dataset):
    """

    :param dataset: the dataset
    :return:
    """


def main():
    """

    main method

    :return:
    """

    dataset = gen_dataset(os.path.join("Data",
                                       "Meeting namer",
                                       "Training Data"))

    for x, y in dataset:
        print(x.shape)
        print(y.shape)
        break


if __name__ == "__main__":
    main()
