"""

author: Arthur wesley

"""

import os

import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

from src import constants
from src.Models.Text_Recognition import trainer, text_utils


def main():
    """

    main testing method

    :return:
    """

    vocab = trainer.get_model_vocab()
    vocab_reverse = text_utils.reverse_vocab(vocab)

    model = load_model(constants.text_recognition)

    image = img_to_array(load_img(os.path.join("Data",
                                               "Meeting Identifier",
                                               "Training Data",
                                               "ext",
                                               "BL-DED-Brizzyne-842123954-2317.jpg")))

    # "  335433.."
    # "  335433.."

    image = image.reshape((1,) + image.shape)

    result = np.argmax(model.predict(image), axis=2)
    text = "".join([vocab_reverse[char] for char in result[0]])

    print(text)
    print(model.predict(image))


if __name__ == "__main__":
    main()
