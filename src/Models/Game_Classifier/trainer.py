"""

Author: Arthur Wesley

"""

from src import constants

from tensorflow import config
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.compiler import mlcompute

from src.Models.Game_Classifier import initalizer


def train_model(dataset):
    """

    creates and trains a model on a limited number of training examples

    :param dataset: dataset to train on
    :return: trained model
    """

    # clear the session so that we can train more than one model
    K.clear_session()

    # initialize the model
    model = initalizer.init_nn()

    # fit the model
    model.fit(dataset, epochs=40)

    return model


def main():
    """

    main method

    :return: None
    """

    # Select CPU device.
    mlcompute.set_mlc_device(device_name='gpu')
    config.run_functions_eagerly(False)

    # print(os.path.exists("Data/Game Classifier/Training Data"))

    training_data = image_dataset_from_directory("Data/Game Classifier/Training Data",
                                                 image_size=constants.dimensions)
    model = train_model(training_data)

    test_data = image_dataset_from_directory("Data/Game Classifier/Test Data",
                                             image_size=constants.dimensions)

    model.evaluate(test_data)
    model.save(constants.game_classifier)


if __name__ == "__main__":
    main()
