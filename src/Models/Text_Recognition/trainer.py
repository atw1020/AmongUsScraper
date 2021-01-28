"""

Author: Arthur Wesley

"""

import os

from tensorflow.keras.preprocessing import image_dataset_from_directory


def get_labels(directory):
    """

    gets the string labels from the specified

    :param directory:
    :return:
    """

    files = os.listdir(directory)

    return directory


def gen_dataset(directory):
    """

    generate a dataset from the

    :param directory:
    :return:
    """

    labels = get_labels(directory)

    return image_dataset_from_directory(directory,
                                        labels=labels)


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

    train_model(None)


if __name__ == "__main__":
    main()
