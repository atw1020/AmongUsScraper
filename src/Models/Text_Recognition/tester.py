"""

author: Arthur wesley

"""

import os

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from src import constants
from src.Models.Text_Recognition import trainer, text_utils, data_generator


def length_accuracy(dataset):
    """

    generates the lengthwise accuracy of a dataset

    :param dataset: dataset to consider
    :return:
    """

    model = load_model(constants.text_recognition)

    for x, y in dataset:
        # create the accuracy evaluation object
        accuracy = SparseCategoricalAccuracy()

        # make a prediction and update the state of the accuracy using it
        prediction = model.predict(x)

        accuracy.update_state(y, prediction)

        print(y[0].numpy())
        print(np.argmax(prediction, axis=-1)[0])

        print("sequences of length", x[1].shape[1] - 1,
              "had an accuracy of", accuracy.result().numpy())


def main():
    """

    main testing method

    :return:
    """

    vocab = trainer.get_model_vocab()

    training_data = data_generator.gen_dataset(os.path.join("Data",
                                                            "Meeting Identifier",
                                                            "Training Data"),
                                               vocab=vocab,
                                               shuffle=False)

    length_accuracy(training_data)

    model = load_model(constants.text_recognition)
    model.evaluate(training_data)


if __name__ == "__main__":
    main()
