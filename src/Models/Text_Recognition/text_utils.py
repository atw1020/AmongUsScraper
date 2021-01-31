"""

Author: Arthur Wesley

"""

import os

import numpy as np

from src import constants


def get_vocab(strings):
    """

    generate a vocabulary form a list of strings

    :param strings: list of strings to get the vocab from
    :return: vocab of the strings
    """

    vocab = sorted(set("".join(strings)))

    return {char: i for i, char in enumerate(vocab)}


def reverse_vocab(vocab):
    """

    reverses a vocabulary dictionary, or more generally, any dictionary

    :param vocab: vocabulary to reverse
    :return: reversed dictionary
    """

    return {v: k for k, v in vocab.items()}


def merge_vocab(vocabs):
    """

    merges

    :param vocabs: vocabularies to merge
    :return:
    """

    vocab = sorted(set([key for vocab in vocabs for key in vocab.keys()]))

    return {char: i for i, char in enumerate(vocab)}


def get_names(directory):
    """

    get the names of all the players in a specified directory

    :param directory: directory to get the data from
    :return: names of all the players
    """

    files = os.listdir(directory)

    # get the names of all the players
    return [file.split("-")[2] for file in files]


def main():
    """

    main testing method

    :return:
    """


if __name__ == "__main__":
    main()
