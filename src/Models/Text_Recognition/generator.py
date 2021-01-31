"""

data generator

"""

import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src import constants
from src.Models.Text_Recognition import text_utils


def name(file):
    """

    generates the name of the

    :param file:
    :return:
    """
    return file.split("-")[2]


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
    files.sort(key=lambda file: len(name(file)), reverse=True)

    # go through every length of string
    for i in range(constants.name_length):

        # go through all of the names that have a length this short

        j = 0
        while j < len(files):

            # make sure that the name is short enough for this group
            if len(name(files[j])) < i:
                break

            # ge the image
            x1 = img_to_array(load_img(os.path.join(directory,
                                                    files[j])))

            # get the characters to feed in
            x2 = text_utils.get_string_input_data(name(files[j])[:i],
                                                  vocab)

            # get the output character
            y = text_utils.get_character_label(name(files[j]),
                                                    i,
                                                    vocab)

            yield (x1, x2), y

            j += 1
            """
            
            expected: 
            
                ((TensorSpec(shape=(45, 220, 3), dtype=tf.int8, name=None), 
                RaggedTensorSpec(TensorShape([None, 60]), tf.int8, 1, tf.int64)), 
                TensorSpec(shape=(60,), dtype=tf.int8, name=None))
                
            got:
            
                ((TensorSpec(shape=(45, 220, 3), dtype=tf.int8, name=None), 
                  TensorSpec(shape=(1, 60), dtype=tf.float64, name=None)), 
                  TensorSpec(shape=(60,), dtype=tf.int8, name=None))
            
            """


def gen_dataset(path, batch_size=32):
    """

    generate a dataset

    :param path: the path to the directory to generate the dataset from
    batch_size: the size of the batch to generate
    :return: dataset
    """

    vocab = text_utils.get_vocab(text_utils.get_names(path))

    vocab_size = len(vocab.keys()) + 2

    dataset = tf.data.Dataset.from_generator(lambda: generator(path, vocab),
                                             output_signature=((tf.TensorSpec(shape=constants.meeting_dimensions + (3,),
                                                                              dtype=tf.int8),
                                                                tf.TensorSpec(shape=(None, 60),
                                                                              dtype=tf.float64)),
                                                               tf.TensorSpec(shape=vocab_size,
                                                                             dtype=tf.int8)))

    return dataset.batch(batch_size)


def main():
    """

    main testing method

    :return:
    """

    gen_dataset(os.path.join("Data",
                             "Meeting Identifier",
                             "Training Data"))


if __name__ == "__main__":
    main()
