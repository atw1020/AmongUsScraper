"""

runs timing tests on the data collector

"""

import time
import random

from src.Data_Collection import data_collector
from src import constants


def collection_parameter_search(limits,
                                reps):
    """



    :param limits: tuple
    :param reps: the number of times to repeat random sampling
    :return:
    """

    step_seed_min, step_seed_max, \
    end_transition_seed_min, end_transition_seed_max = limits

    step_seed_range = step_seed_max - step_seed_min


    for i in range(reps):

        # random number in the specified step seed range
        step_seed = random.random() * step_seed_range + step_seed_min
        step = int(1 / step_seed)

        print("chose a step of", step)

        end_transition_seed = random.randint(end_transition_seed_min, end_transition_seed_max)
        end_transition_step = constants.frames_per_vod / end_transition_seed

        print("chose an end transition step of", end_transition_step)


def main():
    """

    main method

    :return:
    """




if __name__ == "__main__":
    main()
