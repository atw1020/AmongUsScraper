"""

Author: Arthur Wesley

"""

import os
import random

import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array

from src import constants
from src.Models.Text_Recognition import text_utils

total_letters = 0
overlapping_letters = 0


def cords_from_char_int_pair(text):
    """

    generates the integer from a letter that is adjacent to a number (aka a character-
    -integer pair)

    :param text: text containing a character integer pair
    :return: number from the pair
    """

    return int(text[1:])


def gen_label(filename,
              vocab,
              image_dim,
              grid_dimension):
    """

    generate a label for the image based on the name of it

    :param filename: name of the file to get the label from
    :param vocab: vocab to use
    :param image_dim: dimensions of the images
    :param grid_dimension: dimensions
    :return: image tensor
    """

    global total_letters
    global overlapping_letters

    output_channels = 5 + len(vocab)

    step_y = image_dim[0] // grid_dimension[0]
    step_x = image_dim[1] // grid_dimension[1]

    # initialize the output to be all zeros
    output = np.zeros(grid_dimension + (output_channels,),
                      dtype="float64")

    # go through all the letters and set the appropriate values
    letters = filename.split("_")[:-1]

    for letter in letters:

        # split the letters into the items
        items = letter.split("-")

        # get the co-ordinates
        top    = cords_from_char_int_pair(items[1])
        left   = cords_from_char_int_pair(items[2])
        width  = cords_from_char_int_pair(items[3])
        height = cords_from_char_int_pair(items[4])

        # compute the center
        center_x = left + width // 2
        center_y = top + height // 2

        # get the grid co-ordinates of the center
        x = center_x // step_x
        y = center_y // step_y

        # now set the appropriate parameters

        # set PC to 1
        # assert output[y, x, 0] == 0
        if output[y, x, 0] != 0:
            overlapping_letters += 1

        output[y, x, 0] = 1

        # note that all numbers are normalized by the step

        # set the co-ordinates
        output[y, x, 1] = (center_x % step_x) / step_x
        output[y, x, 2] = (center_y % step_y) / step_y

        # set the width and height
        output[y, x, 3] = width / step_x
        output[y, x, 4] = height / step_y

        # get the character ID
        character_id = vocab[items[0]]

        # set the output
        output[y, x, character_id + 5] = 1

        total_letters += 1

    return output


def generator(path,
              vocab,
              shuffle=True,
              image_dim=constants.meeting_dimensions_420p,
              grid_dim=constants.yolo_output_grid_dim,
              verbose=False):
    """

    data generator for images in the specified directory

    :param path: path to the directory that contains the images
    :param vocab: vocabulary of letters to use
    :param shuffle: whether or not to shuffle the dataset
    :param image_dim: dimensions of each image
    :param grid_dim: dimensions of the grid that is placed on the image
    :param verbose: whether or not to give verbose output
    :return:
    """

    global overlapping_letters
    global total_letters

    overlapping_letters = 0
    total_letters = 0

    files = os.listdir(path)

    if ".DS_Store" in files:
        files.remove(".DS_Store")

    if shuffle:
        random.shuffle(files)

    for file in files:

        if verbose:
            print(file)

        x = img_to_array(load_img(os.path.join(path, file)))

        y = gen_label(file,
                      vocab,
                      image_dim,
                      grid_dim)

        yield x, y

    # print a summary
    print("letters overlapped ", overlapping_letters / total_letters, "% of the time", sep="")


def gen_dataset(path,
                vocab,
                batch_size=32,
                shuffle=True,
                image_dim=constants.meeting_dimensions_420p,
                grid_dim=constants.yolo_output_grid_dim,
                verbose=False):
    """

    generate a dataset object for training

    :param path: path to the directory that contains the images
    :param vocab: vocabulary of letters to use
    :param batch_size: the size of the batches the data is divided into
    :param shuffle: whether or not to shuffle the dataset
    :param image_dim: dimensions of each image
    :param grid_dim: dimensions of the grid that is placed on the image
    :param verbose: whether or not to give verbose output
    :return: dataset object for model.fit
    """

    output_channels = 5 + len(vocab)

    dataset = tf.data.Dataset.from_generator(
        lambda: generator(path,
                          vocab,
                          shuffle,
                          image_dim,
                          grid_dim,
                          verbose),
        output_signature=(tf.TensorSpec(shape=image_dim + (3,),
                          dtype=tf.uint8),
                          tf.TensorSpec(shape=grid_dim + (output_channels,),
                          dtype=tf.float64)))

    return dataset.batch(batch_size)


def main():
    """

    main testing method

    :return:
    """

    path = "Data/YOLO/Training Data"
    vocab = text_utils.get_model_vocab()

    dataset = gen_dataset(path,
                          batch_size=1,
                          vocab=vocab,
                          shuffle=False)

    for x, y in dataset:
        save_img("test 3.jpg", x[0])


if __name__ == "__main__":
    main()
