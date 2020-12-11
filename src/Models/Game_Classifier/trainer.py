"""

Author: Arthur Wesley

"""

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image_dataset_from_directory

from src.Models.Game_Classifier import initalizer


def train_model(data_limit):
    """

    creates and trains a model on a limited number of training examples

    :param data_limit: maximum number of training examples (for making learning curves)
    :return: trained model
    """

    # clear the session so that we can train more than one model
    K.clear_session()

    # initialize the model
    model = initalizer.init_nn()

    # initialize the dataset
    dataset = image_dataset_from_directory("Data/Game Classifier/Training Data")

    # fit the model
    model.fit(dataset.take(data_limit), epochs=100)

    return model
