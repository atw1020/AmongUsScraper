"""

Author: Arthur Wesley

"""

import os

from src import constants
from src.Models.Text_Recognition import initalizer
from src.Models.Text_Recognition import text_utils
from src.Models.Text_Recognition import data_generator


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

    training_data = data_generator.gen_dataset(os.path.join("Data",
                                                       "Meeting Identifier",
                                                       "Training Data"),
                                               vocab=vocab)

    test_data = data_generator.gen_dataset(os.path.join("Data",
                                                   "Meeting Identifier",
                                                   "Test Data"),
                                           vocab=vocab)

    # train the model
    model = train_model(training_data, test_data, vocab)
    model.save(constants.text_recognition)

    # 3.0222 at 20 w/ learning rate 0.003
    # 1.5839 at 20 w/ learning rate 0.001
    # 2.2448 at 20 w/ learning rate of 0.0003


if __name__ == "__main__":
    main()
