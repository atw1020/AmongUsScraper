"""

Author: Arthur wesley

"""

import os

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.losses import BinaryCrossentropy

from src.python import constants
from src.python.Models.Winner_Identifier import trainer


def load_image(path):
    """

    load an image as a numpy array

    :param path: path to the image
    :return: numpy array
    """

    image = load_img(path, target_size=constants.dimensions)

    return np.array([img_to_array(image)])


def main():
    """

    main method

    :return:
    """

    filename = "BLPKYLBKWTBNCYLM-844335327-791-37.jpg"

    model = tf.keras.models.load_model(constants.winner_identifier)

    path = os.path.join("Data", "Winner identifier", "Test Data", "ext", filename)

    image = load_image(path)

    prediction = model.predict(image)[0]
    actual = trainer.numpy_from_filename(filename)

    loss = actual * -np.log(prediction) + (1 - actual) * -np.log(1 - prediction)

    print(loss)

    loss = sum(loss) / len(loss)

    print(loss)

    bce = BinaryCrossentropy()
    print(bce(actual, prediction).numpy())


if __name__ == "__main__":
    main()
