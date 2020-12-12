"""

Author: Arthur Wesley

https://gitpython.readthedocs.io/en/stable/search.html?q=branch&check_keywords=yes&area=default

"""

import os

from git.repo.base import Repo

from src.python import constants
from src.python.Models.Game_Classifier import tester


def merge_text(filepath_1, filepath_2):
    """

    merges two text files

    :param filepath_1: path to the first file
    :param filepath_2: path to the second file
    :return:
    """

    base = file_to_dict(filepath_1)
    experimental = file_to_dict(filepath_2)

    # combine the keys into a set
    keys = set(base.keys())
    keys.add(set(experimental.keys()))

    keys = sorted(list(keys))

    # combine the dictionaries
    for key in keys:

        if key not in base:
            # add in blanks for the base if missing
            base[key] = [key] + ["", ""] + experimental[key]
        elif key not in experimental:
            # add in blanks for the experimental if missing
            base[key] = [key] + base[key] + ["", ""]
        else:
            # merge the lines
            base[key] = [key] + base[key] + experimental[key]

        # if we find the first row, store it for later
        if key.isalpha():
            first_row = key

    # put the key into the first position
    keys.remove(first_row)
    keys.insert(0, first_row)

    # write the output document
    with open("output.txt") as file:
        for key in keys:

            # write the line
            file.write(constants.delimiter.join(base[key]))

            # write the newline
            file.write("\n")


def file_to_dict(filepath):
    """

    tokenizes each line by coma space parse and create a dictionary that maps from the first item
    to the remaining items

    :param filepath: path to the file
    :return: dictionary that maps from the first token to a list of the remaining tokens
    """

    result = dict()

    with open(filepath) as file:

        # go through all the lines of the file

        for line in file:

            # tokenize the line
            line = line.strip()
            line = line.split(constants.delimiter)

            # add the line to the dictionary
            result[line[0]] = line[1:]

    return result


def main():
    """

    main method

    :return:
    """

    # get the current repository and the current branch
    repo = Repo("")
    branch = repo.active_branch

    base = "master" + constants.learning_curve_extension
    experimental = branch + constants.learning_curve_extension

    # run learning curves on this branch
    if not os.path.exists(experimental):
        tester.compute_learning_curves(branch)

    # if the master file exists, merge the files
    if os.path.exists(base):

        # check to see if we have the stats from this branch

        # merge the files
        merge_text(base, experimental)

    else:
        raise FileNotFoundError("Could not locate master data, switch to master and compute the learning"
                                "curves then try to merge")


if __name__ == "__main__":
    main()
