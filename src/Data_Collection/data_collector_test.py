"""

runs timing tests on the data collector

"""

import time as t
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

    vods = [
        "874833883",
        "887962963",
        "887579689",
        "887571311",
        "886173458"
    ]

    print("step, end transition step, games found, time elapsed")

    for i in range(reps):

        # random number in the specified step seed range
        step_seed = random.random() * step_seed_range + step_seed_min
        step = int(1 / step_seed)

        end_transition_seed = random.randint(end_transition_seed_min, end_transition_seed_max)
        end_transition_step = int(constants.frames_per_vod / end_transition_seed)

        # set up accumulators
        games_found = 0
        time_elapsed = 0

        # go through all of the test games
        for vod in vods:

            t0 = t.time()

            # create a data collector object and get the winners
            collector = data_collector.DataCollector(vod,
                                                     step=step,
                                                     end_transition_step=end_transition_step,
                                                     verbose=False)

            winners = collector.get_winners()

            t1 = t.time()

            games_found += len(winners)
            time_elapsed += t1 - t0

        # print the results
        print(step, end=", ")
        print(end_transition_step, end=", ")
        print(games_found, end=", ")
        print(time_elapsed)


def main():
    """

    main method

    :return:
    """

    collection_parameter_search((0.05, 0.2, 3, 10),
                                10)

    # ranges searched
    # 1) 0 - 0.5
    #    1 - 10
    # 2) 0.05 - 0.2
    #    3 - 10


if __name__ == "__main__":
    main()
