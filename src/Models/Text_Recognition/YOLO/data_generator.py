"""

Author: Arthur Wesley

"""

import os

import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src import constants
from src.Models.Text_Recognition import text_utils


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

    output_channels = 5 + len(vocab)

    step_y = image_dim[0] // grid_dimension[0]
    step_x = image_dim[1] // grid_dimension[1]

    print(step_x)
    print(step_y)

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
        assert output[y, x, 0] == 0

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

    return output


def generator(path,
              vocab,
              image_dim=constants.meeting_dimensions_420p,
              grid_dim=constants.yolo_output_grid_dim):
    """

    data generator for images in the specified directory

    :param path: path to the directory that contains the images
    :param vocab: vocabulary of letters to use
    :param image_dim: dimensions of each image
    :param grid_dim: dimensions of the grid that is placed on the image
    :return:
    """

    files = os.listdir(path)

    if ".DS_Store" in files:
        files.remove(".DS_Store")

    for file in files:

        x = img_to_array(load_img(os.path.join(path, file)))

        y = gen_label(file,
                      vocab,
                      image_dim,
                      grid_dim)

        yield x, y


def main():
    """

    main testing method

    :return:
    """

    vocab = text_utils.get_model_vocab()
    print(vocab)

    for x, y in generator("Data/YOLO/Training Data", vocab):
        print(x.shape)
        print(y.shape)
        break


if __name__ == "__main__":
    main()
