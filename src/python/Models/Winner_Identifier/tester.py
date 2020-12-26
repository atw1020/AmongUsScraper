"""

Author: Arthur wesley

"""

import os

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src.python import constants


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

    model = tf.keras.models.load_model("Winner Identifier.h5")

    path = os.path.join("Data", "Winner identifier", "Training Data", "ext", "GNPK-838252205-687.jpg")

    image = load_image(path)

    print(model.predict(image))


if __name__ == "__main__":
    main()
