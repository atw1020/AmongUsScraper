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

    for input_data, output_data in dataset:
        break

    predictions = model.predict(input_data)

    # go through the images
    M, H, V, O = predictions.shape

    # go through all the training examples
    for i in range(M):

        # reset the found points
        found_points = []

        for j in range(H):
            for k in range(V):

                if output_data[i, j, k, 0] > constants.image_detection_dropoff:
                    found_points.append((predictions[i, j, k, 0], (j, k), predictions[i, j, k]))

        # sort the points by the probability
        found_points.sort(key=lambda x: x[0], reverse=True)

        # go through all of the letter points
        index = 0
        while index < len(found_points):
            current_box = found_points[index][2][1:5]
            x, y = found_points[index][1]
            actual_box = output_data[i, x, y][1:5]
            print("=" * 50)
            print("the probability of this box being a letter was")
            print(found_points[index][0])
            print("the correct box was")
            print(actual_box.numpy())
            print("the predicted box was")
            print(current_box)

            index += 1


def main():
    """

    main testing method

    :return:
    """

    vocab = text_utils.get_model_vocab()

    dataset = data_generator.gen_dataset("Data/YOLO/Training Data",
                                         vocab,
                                         batch_size=1,
                                         shuffle=False)

    get_letters(dataset.take(1), load())


if __name__ == "__main__":
    main()

