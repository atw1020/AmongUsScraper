"""

Author: Arthur Wesley

https://gitpython.readthedocs.io/en/stable/search.html?q=branch&check_keywords=yes&area=default

"""

import os

from git.repo.base import Repo

from src import constants
from src.Models.Game_Classifier import tester


def merge_text(file_1, file_2):
    """



    :param file_1:
    :param file_2:
    :return:
    """


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

    # check if we have the stats from master

    if os.path.exists(base):

        # check to see if we have the stats from this branch

        if not os.path.exists(experimental):
            tester.compute_learning_curves(branch)

        # merge the files
        merge_text(base, experimental)

    else:
        raise FileNotFoundError("Could not locate master data, switch to master and compute the learning"
                                "curves then try to merge")


if __name__ == "__main__":
    main()
