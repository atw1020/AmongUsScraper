"""

Author: Arthur Wesley

"""

import os

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src import constants
from src.Models.Text_Recognition.trainer import get_model_vocab
from src.Models.Text_Recognition import text_utils


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

    # initialize the text tensor
    text_tensor = np.zeros((1, 1, vocab_size))
    text_tensor[0][0][-2] = 1

    # load the model
    model = load_model(constants.text_recognition)

    # loop through the characters
    while text_tensor[0][-1][-1] != 1:

        # get the next character
        char_index = np.argmax(model.predict([image, text_tensor]))

        next_char = np.zeros((1, 1, vocab_size))
        next_char[0][0][char_index] = 1

        # concatenate the new character
        text_tensor = np.concatenate([text_tensor, next_char],
                                     axis=1)

    # get the specific characters using argmax
    chars = np.argmax(text_tensor, axis=2)[0][1:-1]

    # reverse the vocab
    reverse_vocab = text_utils.reverse_vocab(vocab)

    return "".join([reverse_vocab[char] for char in chars])


def main():
    """

    main method

    :return:
    """

    vocab = get_model_vocab()

    text = read_img(os.path.join("Data",
                                 "Meeting Identifier",
                                 "Training Data",
                                 "CY-LIV-Blaustoise-845808011-3585.jpg"), vocab)

    print('outputs "', text, '"', sep="")


if __name__ == "__main__":
    main()