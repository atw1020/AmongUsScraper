"""

Author: Arthur Wesley

"""

from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import load_img


def get_letters(img_path):
    """

    generate the letters that appear in on the img_path

    :param img_path: the path to the image
    :return: a string from the image
    """

    image = load_img(img_path)

