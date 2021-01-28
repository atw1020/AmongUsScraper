"""

Author: Arthur Wesley

"""

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


def pad_string(st):
    """

    pad the string with spaces

    :param st: string to pad
    :return: padded string
    """

    if len(st) > 10:
        print(st)

    assert len(st) <= 10

    return st + " " * (constants.name_length - len(st))


def label_from_string(st, vocab):
    """

    generates a numpy label containing

    :param st: string to parametrize
    :param vocab: vocabulary
    :return:
    """

    return np.array([vocab[char] for char in pad_string(st)])


def main():
    """

    main testing method

    :return:
    """

    print(pad_string("street "), ":D", sep="")


if __name__ == "__main__":
    main()
