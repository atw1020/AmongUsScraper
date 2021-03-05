"""

Author: Arthur Wesley

"""

import os

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import save_img

from src import constants
from src.Models.Text_Recognition import text_utils
from src.Models.Text_Recognition.YOLO import box_geometry
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
                vocab,
                model,
                image_shape=constants.meeting_dimensions_420p):
    """

    generate the letters that appear in on the img_path

    dataset to get the letters from
    :param dataset: dataset to get the letters for
    :param vocab: vocabulary to get the letters from
    :param model: model to use
    :param image_shape: the dimensions of the images
    :return: a string from the image
    """

    vocab = text_utils.reverse_vocab(vocab)

    predictions = model.predict(dataset)
    images = [x for x, y in dataset]

    # go through the images
    M, V, H, O = predictions.shape

    x_step = image_shape[1] / H
    y_step = image_shape[0] / V

    names = []

    # go through all the training examples
    for i in range(M):

        # save a greyscale image
        """greyscale = predictions[i, :, :, 0].reshape((V, H, 1))
        save_img("greyscale.jpg", greyscale)"""

        save_img("test 2.jpg", images[i][0])

        # reset the found points
        found_boxes = []

        for j in range(V):
            for k in range(H):

                if predictions[i, j, k, 0] > constants.image_detection_dropoff:
                    found_boxes.append((predictions[i, j, k, 0], (j, k), predictions[i, j, k]))

        # sort the points by the probability
        found_boxes.sort(key=lambda x: x[0], reverse=True)

        # go through all of the letter points
        index = 0
        while index < len(found_boxes):

            # unpack the first box
            x_rel, y_rel, w_rel, h_rel = found_boxes[index][2][1:5]
            x, y = found_boxes[index][1]

            # get the absolute co-ordinates and absolute width and height
            x1 = (x + x_rel) * x_step
            y1 = (y + y_rel) * y_step

            w1 = w_rel * x_step
            h1 = h_rel * y_step

            box_1 = (x1, y1, w1, h1)

            # go through all of the remaining points
            for second_box in found_boxes[i + 1:]:

                # unpack the second box
                x_rel, y_rel, w_rel, h_rel = second_box[2][1:5]
                x, y = second_box[1]

                # compute the absolute co-ords
                x2 = (x + x_rel) * x_step
                y2 = (y + y_rel) * y_step

                w2 = w_rel * x_step
                h2 = h_rel * y_step

                box_2 = (x2, y2, w2, h2)

                IoU = box_geometry.IoU(box_1, box_2)

                if IoU > constants.IoU_threshold:
                    found_boxes.remove(second_box)

            index += 1

        letters = []

        # get each letter
        for box in found_boxes:

            # get the letter
            letter = vocab[np.argmax(box[2][5:])]

            # append a tuple of the character and the x co-ord of the letter
            letters.append((letter, box[1][1]))

        # sort the letters
        letters.sort(key=lambda x: x[1])

        # remove the co-ordinates used for ordering
        name = "".join([char for char, x in letters])

        print(name)
        names.append(name)

    return names


def main():
    """

    main testing method

    :return:
    """

    vocab = text_utils.get_model_vocab()

    dataset = data_generator.gen_dataset("Data/YOLO/Training Data",
                                         vocab,
                                         batch_size=1,
                                         shuffle=False,
                                         verbose=False)

    get_letters(dataset.take(1),
                vocab,
                load())


if __name__ == "__main__":
    main()

