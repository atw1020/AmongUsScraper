"""

Author: Arthur Wesley

"""

import os

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src import constants
from src.Models.Text_Recognition.trainer import get_model_vocab


def read_img(img_path, vocab):
    """

    read text from an image

    :param img_path: path to the image
    :param vocab: vocabulary to translate
    :return: string of the player name in the image
    """

    # load the image
    image = img_to_array(load_img(img_path))
    image = image.reshape((1,) + image.shape)

    vocab_size = len(vocab.keys()) + 2

    print(vocab)
    print(vocab_size)

    # initialize the text tensor
    text_tensor = np.zeros((1, vocab_size))
    text_tensor[0][-2] = 1

    # load the model
    model = load_model(constants.text_recognition)
    model.summary()

    # loop through the characters
    while text_tensor[-1][-1] != 1:

        print(image.shape)
        print(text_tensor.shape)

        # get the next character
        next_char = model.predict([image, text_tensor])

        # concatenate the new character
        text_tensor = np.concatenate([text_tensor, next_char], axis=0)

        print(text_tensor.shape)


def main():
    """

    main method

    :return:
    """

    vocab = get_model_vocab()

    read_img(os.path.join("Data",
                          "Meeting Identifier",
                          "Training Data",
                          "BL-DED-Deluxe 4-840885731-1506.jpg"), vocab)


if __name__ == "__main__":
    main()
