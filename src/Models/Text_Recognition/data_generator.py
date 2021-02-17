"""

data generator

"""

import os

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


def get_dataset_sizes(directory):
    """

    get the sizes of the different lengthed datasets in a directory

    :param directory: directory to get the names from
    :return: list that maps from length of items to number of training examples
    """

    files = os.listdir(directory)

    if ".DS_Store" in files:
        files.remove(".DS_Store")

    sizes = [0 for i in range(constants.name_length)]

    for file in files:
        sizes[len(name_from_filepath(file)) - 1] += 1

    return sizes


def generator(directory, length, vocab):
    """

    generator, yields training data and test data

    :param directory: directory to get the images from
    :param length: length of the substrings to generate a dataset for
    :param vocab: vocabulary to use
    :return:
    """

    # get a list of all of the names
    files = os.listdir(directory)

    if ".DS_Store" in files:
        files.remove(".DS_Store")

    # sort the files by name length
    files.sort(key=lambda file: len(name_from_filepath(file)), reverse=True)

    # go through all of the names that have a length this short

    j = 0
    while j < len(files):

        if length != len(name_from_filepath(files[j])) and length is not None:
            j += 1
            continue

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


def random_generator(directory, length, vocab):
    """

    generate a random dataset

    :param directory: pacified argument to shadow generator()
    :param length: length of the sequences
    :param vocab: vocabulary to use
    :return:
    """

    vocab_size = len(vocab.keys()) + 2

    for i in range(5):

        # yield random data
        x1 = tf.random.uniform(shape=constants.meeting_dimensions_420p + (3,),
                               minval=0, maxval=256,
                               dtype=tf.dtypes.int32)
        x2 = tf.random.uniform(shape=(length + 1,),
                               minval=0, maxval=64,
                               dtype=tf.dtypes.int32)

        y = tf.random.uniform(shape=(length + 1,),
                              minval=0, maxval=vocab_size,
                              dtype=tf.dtypes.int32)

        yield (x1, x2), y


def gen_dataset_batchless(path,
                          length,
                          vocab,
                          batch_size,
                          max_batch_size,
                          input_dim,
                          generator_func=generator):
    """

    generate a dataset

    :param path: the path to the directory to generate the dataset from
    :param length: length of the strings to generate the dataset for
    :param vocab: vocabulary to use
    :param batch_size: size of the batches to divide the dataset into
    :param max_batch_size: maximum size of this batch
    :param input_dim: the dimension of the input images
    :param generator_func: generator function to use (replaced for apple silicon testing)
    :return: dataset
    """

    dataset = tf.data.Dataset.from_generator(
        lambda: generator_func(path, length, vocab),
        output_signature=(
            (
                tf.TensorSpec(shape=input_dim + (3,),
                              dtype=tf.int8),
                tf.TensorSpec(shape=(None,),
                              dtype=tf.float64)
            ),
            tf.TensorSpec(shape=(None,),
                          dtype=tf.int8))
    )

    # now that we have the dataset, split it into many datasets where each contains inputs
    # of the same length

    if batch_size is None:
        return dataset.batch(max_batch_size)
    else:
        return dataset.batch(batch_size)


def gen_dataset(path,
                batch_size=32,
                vocab=None,
                shuffle=True,
                input_dim=constants.meeting_dimensions,
                random_dataset=False):
    """

    generate a dataset in batches

    :param path: path to the directory to get the dataset from
    :param batch_size: size of the batches to generate
    :param vocab: vocabulary to use
    :param shuffle: whether or not to shuffle the dataset
    :param input_dim: dimension of the input images
    :param random_dataset: whether or not to use a random dataset
    :return: dataset with batches
    """

    if random_dataset:
        generator_func = random_generator
    else:
        generator_func = generator

    if vocab is None:
        vocab = text_utils.get_vocab(text_utils.get_names(path))

    max_batch_sizes = get_dataset_sizes(path)

    datasets = [gen_dataset_batchless(path,
                                      i + 1,
                                      vocab,
                                      batch_size,
                                      max_batch_sizes[i],
                                      input_dim,
                                      generator_func=generator_func)
                for i in range(constants.name_length)]

    # concatenate the datasets
    dataset = datasets[0]

    for ds in datasets[1:]:
        dataset = dataset.concatenate(ds)

    # shuffle the dataset

    if shuffle:
        return dataset.shuffle(buffer_size=1000)
    else:
        return dataset


def main():
    """

    main testing method

    :return:
    """

    path = os.path.join("Data",
                        "Meeting Identifier",
                        "High Res Training Data")

    training_data = gen_dataset(os.path.join("Data",
                                             "Meeting Identifier",
                                             "Training Data"),
                                random_dataset=True,
                                input_dim=constants.meeting_dimensions_420p)

    for (x1, x2), y in training_data:

        print("=" * 60)

        print(x1.shape)
        print(x2.shape)

        print(y.shape)


if __name__ == "__main__":
    main()
