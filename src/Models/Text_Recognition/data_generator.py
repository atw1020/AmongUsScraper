"""

data generator

"""

import os

import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src import constants
from src.Models.Text_Recognition import text_utils


def name_from_filepath(file):
    """

    generates the name of the

    :param file:
    :return:
    """
    return file.split("-")[2]


def get_name_length_indices(directory):
    """

    gets the indices at which the lengths of names changes

    :param directory: directory to get the names from
    :return: indices at which the length of strings changes
    """

    # get a list of all of the names
    names = [name_from_filepath(file) for file in os.listdir(directory)]

    names.sort(key=len, reverse=True)

    # get the indices
    indices = [0]

    current_len = len(names[0])

    for i, name in enumerate(names):

        if len(name) != current_len:
            # update the current length
            current_len = len(name)
            indices.append(i)

    return indices


def generator(directory, vocab):
    """

    generator, yields training data and test data

    :param directory: directory to get the images from
    :param vocab: vocabulary to use
    :return:
    """

    # get a list of all of the names
    files = os.listdir(directory)

    # sort the files by name length
    files.sort(key=lambda file: len(name_from_filepath(file)), reverse=True)

    # go through all of the names that have a length this short

    j = 0
    while j < len(files):

        # ge the image
        x1 = img_to_array(load_img(os.path.join(directory,
                                                files[j])))

        # get the characters to feed in
        x2 = text_utils.get_string_input_data(name_from_filepath(files[j]),
                                              vocab)

        # get the output character
        y = text_utils.get_character_label(name_from_filepath(files[j]),
                                           vocab)

        yield (x1, x2), y

        j += 1


def gen_dataset(path,
                vocab,
                batch_size=32):
    """

    generate a dataset

    :param path: the path to the directory to generate the dataset from
    :param vocab: vocabulary to use
    :param batch_size: size of the batches to divide the dataset into
    :return: dataset
    """

    dataset = tf.data.Dataset.from_generator(lambda: generator(path, vocab),
                                             output_signature=((tf.TensorSpec(shape=constants.meeting_dimensions + (3,),
                                                                              dtype=tf.int8),
                                                                tf.TensorSpec(shape=(None,),
                                                                              dtype=tf.float64)),
                                                               tf.TensorSpec(shape=(None,),
                                                                             dtype=tf.int8)))

    # now that we have the dataset, split it into many datasets where each contains inputs
    # of the same length

    return dataset.batch(batch_size)


def main():
    """

    main testing method

    :return:
    """

    path = os.path.join("Data",
                        "Meeting Identifier",
                        "Training Data")

    dataset = gen_dataset(path)

    for (x1, x2), y in dataset:
        print(x2)
        print(y)
        break


if __name__ == "__main__":
    main()
