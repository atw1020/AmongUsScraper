"""

Author: Arthur Wesley

"""

import os

import numpy as np


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


def get_string_input_data(st, vocab):
    """

    get a set of input data from a string given a vocab

    :param st: string to generate
    :param vocab: vocabulary to use
    :return: numpy array representing the string
    """

    # add two to the vocab size for the start and end characters
    vocab_size = len(vocab.keys()) + 2

    return np.array([vocab_size - 2] + [vocab[char] for char in st])


def get_character_label(st, vocab):
    """

    gets a character's label from the specified vocabulary

    :param st: string to get the label from
    :param index: index of the character to get the label for
    :param vocab: vocabulary to create the label using
    :return: character label
    """

    vocab_size = len(vocab.keys()) + 2

    return [vocab[char] for char in st] + [vocab_size - 1]


def main():
    """

    main testing method

    :return:
    """


if __name__ == "__main__":
    main()
