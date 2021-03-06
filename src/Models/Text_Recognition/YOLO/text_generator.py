"""

Author: Arthur Wesley

"""

import os
import sys

import numpy as np

import tensorflow as tf
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

    color = (21, 53, 232)

    # convert tensor into numpy array
    input_image = input_image.numpy()

    # get the image dimensions
    H, W, C = input_image.shape

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
            input_image[min(int(t), H - 1), min(int(i), W - 1)] = color

        # color the left
        for i in range(int(t), int(t + h)):
            input_image[min(int(i), H - 1), min(int(l), W - 1)] = color

        # color the bottom
        for i in range(int(l), int(l + w)):
            input_image[min(int(t + h), H - 1), min(int(i), W - 1)] = color

        # color the right
        for i in range(int(t), int(t + h)):
            input_image[min(int(i), H - 1), min(int(l + w), W - 1)] = color

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
    output_channels = 5 + len(vocab)

    # predictions = model.predict(dataset)
    images = [x for x, y in dataset]
    predictions = np.array([y.numpy()[0] for x, y in dataset])
    y_true = predictions

    # go through the images
    M, V, H, O = predictions.shape

    x_step = image_shape[1] // H
    y_step = image_shape[0] // V

    names = []

    # go through all the training examples
    for i in range(M):

        # save a greyscale image
        print(y_true.shape)
        # greyscale = y_true[i, :, :, output_channels].reshape((V, H, 1))

        for j in range(1):  # constants.anchor_boxes):

            # save a greyscale image
            greyscale = predictions[i, :, :, j * output_channels].reshape((V, H, 1))

            # set the upper left corner as a reference
            greyscale[0, 0] = 1

            # save the image
            save_img("greyscale predictions" + str(j) + ".jpg", greyscale)

            greyscale[0, 0] = 0

            greyscale = y_true[i, :, :, j * output_channels].reshape((V, H, 1))
            save_img("greyscale true" + str(j) + ".jpg", greyscale)

        # reset the found points
        found_boxes = []

        # probabilities = sorted(list(predictions[i, :, :, ::output_channels].flatten()), reverse=True)
        # print(probabilities[:10])

        # go through all of the rows and columns of the predictions

        for k in range(H):
            for j in range(V):

                for l in range(constants.anchor_boxes):

                    if predictions[i, j, k, l * output_channels] > constants.image_detection_dropoff:
                        found_boxes.append((predictions[i, j, k, l * output_channels],
                                            (j, k),
                                            predictions[i, j, k, l * output_channels: (l + 1) * output_channels]))

        # sort the points by the probability
        found_boxes.sort(key=lambda x: x[0], reverse=True)
        # print(len(found_boxes))

        # get rid of all boxes with a high IoU (intersection over union)
        index = 0
        while index < len(found_boxes):

            # unpack the first box
            x_rel, y_rel, w_rel, h_rel = found_boxes[index][2][1:5]
            y, x = found_boxes[index][1]

            # get the absolute co-ordinates and absolute width and height
            x1 = (x + x_rel) * x_step
            y1 = (y + y_rel) * y_step

            w1 = w_rel * x_step
            h1 = h_rel * y_step

            box_1 = (x1, y1, w1, h1)

            print("box 1 was", box_1)

            # go through all of the remaining points
            for j, second_box in enumerate(found_boxes[index + 1:]):

                # unpack the second box
                x_rel, y_rel, w_rel, h_rel = second_box[2][1:5]
                y, x = second_box[1]

                # compute the absolute co-ords
                x2 = (x + x_rel) * x_step
                y2 = (y + y_rel) * y_step

                w2 = w_rel * x_step
                h2 = h_rel * y_step

                box_2 = (x2, y2, w2, h2)

                print("box 2 was", box_2)

                IoU = box_geometry.IoU(box_1, box_2)

                print(IoU)

                if IoU > constants.IoU_threshold:
                    del found_boxes[index + 1 + j]

            index += 1

        # print(len(found_boxes))

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
                                         batch_size=100,
                                         verbose=False,
                                         shuffle=False,
                                         image_dim=constants.meeting_dimensions_420p)

    get_letters(dataset.take(1),
                vocab,
                load())

    """model = load()

    for x, y in dataset.take(2):
        y_pred = model(x)

        pc_loss, mse_loss = model.loss.loss_summary(y, y_pred)

        pc_loss = tf.reduce_mean(pc_loss).numpy()
        mse_loss = tf.reduce_mean(mse_loss).numpy()

        print(pc_loss)
        print(mse_loss)"""


if __name__ == "__main__":
    main()

