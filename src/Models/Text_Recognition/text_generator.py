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
    text_tensor = np.array([[vocab_size - 2]])

    # load the model
    model = load_model(constants.text_recognition)

    # loop through the characters
    while text_tensor[0][-1] != vocab_size - 1:

        # get the next character
        char = np.argmax(model.predict([image, text_tensor])[:, -1])

        next_char = np.array([[char]])

        # concatenate the new character
        text_tensor = np.concatenate([text_tensor, next_char],
                                     axis=1)

    # get the specific characters using argmax
    chars = text_tensor[0][1:-1]
    print(chars)

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
                                 "PR-DED-Pengwin-875385223-800.jpg"), vocab)

    print('outputs "', text, '"', sep="")


if __name__ == "__main__":
    main()
