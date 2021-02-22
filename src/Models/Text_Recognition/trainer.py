"""

Author: Arthur Wesley

"""

import os
import copy
import time as t

import tensorflow as tf

from tensorflow import config
from tensorflow.keras.callbacks import Callback
from tensorflow.python.compiler import mlcompute

from kerastuner.tuners import BayesianOptimization
from kerastuner import HyperParameters

from src import constants
from src.Models.Text_Recognition.fit import ModelFitter
from src.Models.Text_Recognition import initalizer, text_utils, data_generator


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
                vocab,
                resolution=constants.meeting_dimensions):
    """

    train a model on the specified dataset

    :param training_data: the dataset to train on
    :param test_data: validation data
    :param vocab: vocabulary to use
    :param resolution: resolution of the network to train
    :return: trained model
    """

    model = initalizer.init_nn(vocab,
                               image_dimensions=resolution)
    model.summary()

    cb = TrueAccuracyCallback(training_data)

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
    high_res_vocab = get_vocab(os.path.join("Data",
                                            "Meeting Identifier",
                                            "High Res Training Data"))

    return text_utils.merge_vocab((train_vocab, test_vocab, high_res_vocab))


class TrueAccuracyCallback(Callback):

    def __init__(self, training_data):
        """

        initalize the callback

        :param training_data: training data to evaluate
        """
        super().__init__()

        # initialize the training data
        self.training_data = training_data

    def __copy__(self):
        return type(self)(self.training_data)

    def __deepcopy__(self, memodict={}):
        id_self = id(self)
        _copy = memodict[id_self]

        if _copy is None:
            _copy = type(self)(copy.deepcopy(self.training_data, memodict))
            memodict[id_self] = _copy

        return _copy

    def on_epoch_end(self, epoch, logs=None):
        """

        print the true accuracy at the end of the epoch

        :param epoch: current epoch
        :param logs: data logs from the epoch
        :return: None
        """

        self.model.evaluate(self.training_data)


class TimeHistory(Callback):

    def __init__(self):
        super().__init__()

        self.times = []
        self.epoch_time_start = 0

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = t.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(t.time() - self.epoch_time_start)


def main():
    """

    main method

    :return:
    """

    vocab = get_model_vocab()

    mlcompute.set_mlc_device(device_name="gpu")

    training_data = data_generator.gen_dataset(os.path.join("Data",
                                                            "Meeting Identifier",
                                                            "Reduced High res Training Data"),
                                               input_dim=constants.meeting_dimensions_420p,
                                               # batch_size=None,
                                               vocab=vocab)

    test_data = data_generator.gen_dataset(os.path.join("Data",
                                                        "Meeting Identifier",
                                                        "High Res Test Data"),
                                           input_dim=constants.meeting_dimensions_420p,
                                           vocab=vocab)

    """model = initalizer.init_nn(vocab,
                               image_dimensions=constants.meeting_dimensions_420p)

    model.summary()

    fitter = ModelFitter(model)
    fitter.fit(training_data,
               epochs=300,
               validation_data=test_data)"""

    model = train_model(training_data,
                        test_data,
                        vocab,
                        resolution=constants.meeting_dimensions_420p)
    model.save(constants.text_recognition)

    """tuner = BayesianOptimization(lambda hp: initalizer.init_nn(vocab,
                                                               hp,
                                                               image_dimensions=constants.meeting_dimensions_420p),
                                 objective="accuracy",
                                 max_trials=50,
                                 executions_per_trial=2,
                                 directory="Models",
                                 project_name="Bayesian Text Recognition")

    tuner.search(training_data,
                 epochs=300)"""


if __name__ == "__main__":
    main()
