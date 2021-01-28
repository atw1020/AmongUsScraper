"""

Author: Arthur Wesley

"""

import os

from tensorflow.keras.preprocessing import image_dataset_from_directory

from src.Models.Text_Recognition import text_utils


def string_to_numpy(st, translation_dict):
    """



    :param st:
    :param translation_dict:
    :return:
    """


def get_labels(directory):
    """

    gets the string labels from the specified

    :param directory:
    :return:
    """

    files = os.listdir(os.path.join(directory,
                                    "ext"))

    return [file.split("-")[2] for file in files]


def gen_dataset(directory):
    """

    generate a dataset from the

    :param directory:
    :return:
    """

    labels = get_labels(directory)

    return image_dataset_from_directory(directory,
                                        labels=labels)


def train_model(dataset):
    """

    :param dataset: the dataset
    :return:
    """


def main():
    """

    main method

    :return:
    """

    labels = get_labels(os.path.join("Data",
                                     "Meeting namer",
                                     "Training Data"))

    vocab = text_utils.get_vocab(labels)

    print(text_utils.label_from_string(labels[0], vocab))
    print(labels)


if __name__ == "__main__":
    main()
