"""

Author: Arthur Wesley

"""

import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import save_img

from src import constants
from src.Models.Text_Recognition import text_utils
from src.Models.Text_Recognition.YOLO import data_generator
from src.Models.Text_Recognition.YOLO.loss import YoloLoss
from src.Models.Text_Recognition.YOLO.output_activation import YoloOutput


def load():
    """

    load a model

    :return:
    """

    return load_model(constants.letter_detection,
                      custom_objects={"YoloOutput": YoloOutput,
                                      "YoloLoss": YoloLoss})


def get_letters(dataset,
                model):
    """

    generate the letters that appear in on the img_path

    dataset to get the letters from
    :param dataset: dataset to get the letters for
    :param model: model to use
    :return: a string from the image
    """

    predictions = model.predict(dataset)

    # go through the images
    M, H, V, O = predictions.shape

    # go through all the training examples
    for i in range(M):

        # reset the found points
        found_points = []

        for j in range(H):
            for k in range(V):

                if predictions[i, j, k, 0] > constants.image_detection_dropoff:
                    found_points.append(((j, k), predictions[i, j, k]))

        print(len(found_points))


def main():
    """

    main testing method

    :return:
    """

    vocab = text_utils.get_model_vocab()

    dataset = data_generator.gen_dataset("Data/YOLO/Training Data",
                                         vocab,
                                         batch_size=1)

    get_letters(dataset.take(1), load())


if __name__ == "__main__":
    main()

