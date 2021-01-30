"""

Author: Arthur Wesley

"""

import os

import numpy as np

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src import constants
from src.Models.Text_Recognition import initalizer
from src.Models.Text_Recognition import text_utils


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


def get_labels(directory, vocab):
    """

    gets the string labels from the specified

    :param directory: directory to get the labels from
    :param vocab: vocabulary to use when getting the labels
    :return: labels of the images in that directory and the vocab used for them
    """

    files = os.listdir(os.path.join(directory,
                                    "ext"))

    # get the names of all the players
    names = [file.split("-")[2] for file in files]

    if vocab is None:
        vocab = text_utils.get_vocab(names)

    return [text_utils.label_from_string(name, vocab) for name in names], vocab


def gen_dataset(directory, vocab=None):
    """

    generate a dataset from the

    :param vocab: vocabulary to use
    :param directory: directory to generate the dataset from
    :return:
    """

    labels, vocab = get_labels(directory, vocab)

    return image_dataset_from_directory(directory,
                                        labels=labels,
                                        image_size=constants.meeting_dimensions), vocab


def train_model(dataset, test_data, vocab):
    """

    train a model on the specified dataset

    :param dataset: the dataset to train on
    :param test_data: validation data
    :param vocab: vocabulary to use
    :return: trained model
    """

    model = initalizer.init_nn(vocab)

    model.fit(dataset,
              validation_data=test_data,
              epochs=5)

    return model


def get_model_vocab():
    """



    :return:
    """

    # get the vocabularies
    train_vocab = get_vocab(os.path.join("Data",
                                         "Meeting namer",
                                         "Training Data"))
    test_vocab = get_vocab(os.path.join("Data",
                                        "Meeting namer",
                                        "Test Data"))

    return text_utils.merge_vocab((train_vocab, test_vocab))


def main():
    """

    main method

    :return:
    """

    vocab = get_model_vocab()

    # get the datasets
    training_data, vocab = gen_dataset(os.path.join("Data",
                                                    "Meeting namer",
                                                    "Test Data"), vocab)
    test_data, vocab = gen_dataset(os.path.join("Data",
                                                "Meeting namer",
                                                "Test Data"), vocab)

    # train the model
    model = train_model(training_data, test_data, vocab)
    model.save(constants.text_recognition)


if __name__ == "__main__":
    main()
