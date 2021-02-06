"""

Author: Arthur Wesley

"""

import os

from kerastuner.tuners import BayesianOptimization
from kerastuner import HyperParameters

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


def train_model(training_data,
                test_data,
                vocab):
    """

    train a model on the specified dataset

    :param training_data: the dataset to train on
    :param test_data: validation data
    :param vocab: vocabulary to use
    :return: trained model
    """

    model = initalizer.init_nn(vocab)

    model.fit(training_data,
              validation_data=test_data,
              epochs=300)

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
                                               vocab=vocab,
                                               batch_size=1)

    test_data = data_generator.gen_dataset(os.path.join("Data",
                                                        "Meeting Identifier",
                                                        "Test Data"),
                                           vocab=vocab)

    model = train_model(training_data, test_data, vocab)
    model.save(constants.text_recognition)

    tuner = BayesianOptimization(lambda hp: initalizer.init_nn(vocab, hp),
                                 objective="val_accuracy",
                                 max_trials=50,
                                 executions_per_trial=1,
                                 directory="Models",
                                 project_name="Bayesian Text Recognition")

    tuner.search(training_data,
                 epochs=50,
                 validation_data=test_data)


if __name__ == "__main__":
    main()
