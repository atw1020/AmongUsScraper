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


def main():
    """

    main method

    :return:
    """

    training_path = os.path.join("Data",
                                  "Meeting namer",
                                  "Training Data")
    test_path = os.path.join("Data",
                             "Meeting namer",
                             "Test Data")

    # get the vocabularies
    train_vocab = get_vocab(training_path)
    test_vocab = get_vocab(test_path)

    vocab = text_utils.merge_vocab((train_vocab, test_vocab))
    vocab_reverse = text_utils.reverse_vocab(vocab)

    # get the datasets
    training_data, vocab = gen_dataset(training_path, vocab)
    test_data, vocab = gen_dataset(test_path, vocab)

    # train the model
    model = train_model(training_data, test_data, vocab)

    image = img_to_array(load_img(os.path.join("Data",
                                               "Meeting namer",
                                               "Test Data",
                                               "ext",
                                               "OR-DED-Nyxpip-844335327-835.jpg")))

    image = image.reshape((1,) + image.shape)

    result = np.argmax(model.predict(image), axis=2)
    text = "".join([vocab_reverse[char] for char in result])

    print(text)


if __name__ == "__main__":
    main()
