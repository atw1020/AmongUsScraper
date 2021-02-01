"""

author: Arthur wesley

"""

import os

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from src import constants
from src.Models.Text_Recognition import trainer, text_utils, data_generator


def main():
    """

    main testing method

    :return:
    """

    vocab = trainer.get_model_vocab()

    model = load_model(constants.text_recognition)

    training_data = data_generator.gen_dataset(os.path.join("Data",
                                                            "Meeting Identifier",
                                                            "Test Data"),
                                               vocab=vocab,
                                               shuffle=False)

    for x, y in training_data:

        # create the accuracy evaluation object
        accuracy = CategoricalAccuracy()

        # make a prediction and update the state of the accuracy using it
        prediction = model.predict(x)
        accuracy.update_state(y, prediction)

        print("sequences of length", x[1].shape[1],
              "had an accuracy of", accuracy.result().numpy())


if __name__ == "__main__":
    main()
