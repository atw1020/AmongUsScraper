"""

Author: Arthur Wesley

"""

import os

from src import constants
from src.Models.Text_Recognition import initalizer
from src.Models.Text_Recognition import text_utils
from src.Models.Text_Recognition import generator


def get_vocab(directory):
    """

    get the vocab from a directory

    :param directory: directory to get the vocabulary from
    :return: vocabulary from the directory
    """

    names = text_utils.get_names(directory)

    return text_utils.get_vocab(names)


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
              epochs=50)

    return model


def get_model_vocab():
    """



    :return:
    """

    # get the vocabularies
    train_vocab = get_vocab(os.path.join("Data",
                                         "Meeting Identifier",
                                         "Training Data"))
    test_vocab = get_vocab(os.path.join("Data",
                                        "Meeting Identifier",
                                        "Test Data"))

    return text_utils.merge_vocab((train_vocab, test_vocab))


def main():
    """

    main method

    :return:
    """

    vocab = get_model_vocab()

    training_data = generator.generator(os.path.join("Data",
                                                     "Meeting Identifier",
                                                     "Training Data"), vocab)

    # train the model
    model = train_model(training_data, None, vocab)
    model.save(constants.text_recognition)


if __name__ == "__main__":
    main()
