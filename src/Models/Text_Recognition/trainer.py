"""

Author: Arthur Wesley

"""

import os

from tensorflow.keras.callbacks import Callback

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


def train_random_model(training_data,
                       test_data,
                       vocab,
                       repeats=10,
                       automatic=False):
    """

    train a randomly generated model

    :param training_data: dataset to train on
    :param test_data: testing data
    :param vocab: vocabulary
    :param repeats: number of times to repeat the experiment
    :param automatic: whether or not to automaticaly store the results
    :return: None
    """

    hyperparameters = initalizer.get_random_hyperparameters()

    keys = sorted(hyperparameters.keys())
    print(", ".join(keys), "training accuracy", "test accuracy", sep=", ")

    for i in range(repeats):

        while True:
            try:
                model, kwargs = initalizer.init_random_nn(vocab)
                break
            except ValueError:
                continue

        print(", ".join([str(kwargs[key]) for key in keys]))

        # fit the model
        model.fit(training_data,
                  validation_data=test_data,
                  epochs=300)

        training_accuracy = model.evaluate(training_data)[1]
        test_accuracy = model.evaluate(test_data)[1]

        # print the values
        print(", ".join([str(kwargs[key]) for key in keys]),
              training_accuracy,
              test_accuracy,
              sep=", ")

        if not automatic:
            input("process completed, press any key to continue...")

        else:
            # automatically store the results and continue
            with open("src/Models/Text_Recognition/text recognition hyperparameters.txt", "a") as file:

                # write the data
                items = [str(kwargs[key]) for key in keys] + [str(training_accuracy), str(test_accuracy)]

                file.write(", ".join(items))

                # write the newline
                file.write("\n")


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

    cb = TrueAccuracyCallback(training_data)

    model.fit(training_data,
              validation_data=test_data,
              epochs=1000,
              callbacks=[cb])

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

    def on_epoch_end(self, epoch, logs=None):
        """

        print the true accuracy at the end of the epoch

        :param epoch: current epoch
        :param logs: data logs from the epoch
        :return: None
        """

        self.model.evaluate(self.training_data)


def main():
    """

    main method

    :return:
    """

    vocab = get_model_vocab()

    """training_data = data_generator.gen_dataset(os.path.join("Data",
                                                            "Meeting Identifier",
                                                            "Training Data"),
                                               vocab=vocab)

    test_data = data_generator.gen_dataset(os.path.join("Data",
                                                        "Meeting Identifier",
                                                        "Test Data"),
                                           vocab=vocab)"""

    # train the model

    """train_random_model(training_data,
                       test_data,
                       vocab,
                       automatic=True,
                       repeats=50)"""

    high_res_data = data_generator.gen_dataset("Data/Meeting Identifier/Reduced High Res Training Data",
                                               vocab=vocab,
                                               input_dim=constants.meeting_dimensions_420p,
                                               batch_size=None)

    high_res_test_data = data_generator.gen_dataset("Data/Meeting Identifier/Reduced High Res Test Data",
                                                    vocab=vocab,
                                                    input_dim=constants.meeting_dimensions_420p)

    model = train_model(high_res_data,
                        high_res_test_data,
                        vocab,
                        resolution=constants.meeting_dimensions_420p)
    model.save(constants.text_recognition)


if __name__ == "__main__":
    main()
