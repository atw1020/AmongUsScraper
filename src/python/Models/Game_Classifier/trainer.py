"""

Author: Arthur Wesley

"""

from src.python import constants

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image_dataset_from_directory

from src.python.Models.Game_Classifier import initalizer


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

    # print(os.path.exists("Data/Game Classifier/Training Data"))

    training_data = image_dataset_from_directory("Data/Game Classifier/Training Datb",
                                                 image_size=constants.dimensions)
    model = train_model(training_data)

    test_data = image_dataset_from_directory("Data/Game Classifier/Test Data",
                                                 image_size=constants.dimensions)

    model.evaluate(test_data)
    model.save("Game Classifier.h5")


if __name__ == "__main__":
    main()
