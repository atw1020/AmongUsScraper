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


def add_boxes(letters,
              input_image,
              x_step,
              y_step):
    """

    generates a numpy array

    :param letters: list of letters to draw boxes for
    :param input_image: image to put the boxes on
    :param x_step: the horizontal step between output boxes
    :param y_step: the vertical step between output boxes
    :return: numpy image with boxes on it
    """

    # convert tensor into numpy array
    input_image = input_image.numpy()

    for letter in letters:

        # unpack the first box
        x_rel, y_rel, w_rel, h_rel = letter[2]
        y, x = letter[1]

        # get the absolute co-ordinates and absolute width and height
        x = (x + x_rel) * x_step
        y = (y + y_rel) * y_step

        w = w_rel * x_step
        h = h_rel * y_step

        # get the top left corner
        t = y - h / 2
        l = x - w / 2

        # color the top
        for i in range(int(l), int(l + w)):
            input_image[int(t), int(i)] = constants.box_color

        # color the left
        for i in range(int(t), int(t + h)):
            input_image[int(i), int(l)] = constants.box_color

        # color the bottom
        for i in range(int(l), int(l + w)):
            input_image[int(t + h), int(i)] = constants.box_color

        # color the right
        for i in range(int(t), int(t + h)):
            input_image[int(i), int(l + w)] = constants.box_color

    return input_image


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

    # predictions = model.predict(dataset)
    images = [x for x, y in dataset]
    predictions = np.array([y.numpy() for x, y in dataset][0])

    # go through the images
    print(predictions.shape)
    M, V, H, O = predictions.shape

    x_step = image_shape[1] / H
    y_step = image_shape[0] / V

    names = []

    # go through all the training examples
    for i in range(M):

        # save a greyscale image
        greyscale = predictions[i, :, :, 0].reshape((V, H, 1))
        save_img("greyscale.jpg", greyscale)

        # reset the found points
        found_boxes = []

        probabilities = sorted(list(predictions[i, :, :, 0].flatten()), reverse=True)
        print(probabilities[:10])

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
            letters.append((letter, box[1], box[2][1:5]))

        # sort the letters
        letters.sort(key=lambda x: x[1][1])

        image = add_boxes(letters,
                          images[i][0],
                          x_step,
                          y_step)
        save_img("boxed.jpg", image)

        # remove the co-ordinates used for ordering
        name = "".join([char for char, x, box in letters])

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
                                         verbose=False)

    """get_letters(dataset.take(1),
                vocab,
                load())"""

    get_letters(dataset.take(1), vocab, None)


if __name__ == "__main__":
    main()

