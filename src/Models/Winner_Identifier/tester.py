"""

Author: Arthur wesley

"""

import os

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.losses import BinaryCrossentropy

from src import constants
from src.Models.Winner_Identifier import trainer


def load_image(path):
    """

    load an image as a numpy array

    :param path: path to the image
    :return: numpy array
    """

    image = load_img(path, target_size=constants.dimensions)

    return np.array([img_to_array(image)])


def print_predictions(model, filename):
    """



    :param model:
    :param filename:
    :return:
    """

    path = os.path.join("Data", "Winner identifier", "Training Data", "ext", filename)

    image = load_image(path)

    prediction = model.predict(image)[0]
    actual = trainer.numpy_from_filename(filename)

    print("Prediction", prediction)
    print("Actual", actual)


def main():
    """

    main method

    :return:
    """

    model = tf.keras.models.load_model(constants.winner_identifier)

    print_predictions(model, "BKBN-836760118-1044-0.jpg")
    print_predictions(model, "BKPR-854056599-915-250.jpg")

    print_predictions(model, "BLGNORYLWTPRCYLM-856248469-2603-200.jpg")
    print_predictions(model, "BLGNPKORYLBKPRBN-845650806-1000-112.jpg")


if __name__ == "__main__":
    main()
