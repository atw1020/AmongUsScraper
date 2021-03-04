"""

Author: Arthur Wesley

"""

import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img

from src import constants
from src.Models.Text_Recognition.YOLO.output_activation import YoloOutput


def get_letters(img_path, model):
    """

    generate the letters that appear in on the img_path

    :param img_path: the path to the image
    :param model: model to use
    :return: a string from the image
    """

    image = load_img(img_path)

    predictions = model.predict(image)

    # slice the predictions and save
    print(predictions.shape)


def main():
    """

    main testing method

    :return:
    """

    model = load_model(constants.letter_detection,
                       custom_objects={"YoloOutput": YoloOutput})

    files = os.listdir("Data/YOLO/Training Data")

    get_letters(files[0], model)


if __name__ == "__main__":
    main()

