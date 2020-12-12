"""

Author: Arthur Wesley

"""

from tensorflow.keras import backend as K

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
    model.fit(dataset, lr=0.03, epochs=100)

    return model
