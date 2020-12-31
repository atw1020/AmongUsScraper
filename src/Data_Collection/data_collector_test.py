"""

runs timing tests on the data collector

"""

import time

import data_collector


def get_step_data(steps, batch_size=32):
    """

    gets run data from a given number of steps

    :param batch_size: size of each batch
    :param steps: number of steps to take
    :return: None (prints summary)
    """

    transitions = []
    games = []

    t0 = time.time()

    collector = data_collector.DataCollector("825004778",
                                             step=steps,
                                             verbose=False,
                                             batch_size=batch_size)
    transitions = transitions + collector.get_transitions()
    games = games + collector.get_game_transitions()
    del collector
    collector = data_collector.DataCollector("846990283",
                                             step=steps,
                                             verbose=False,
                                             batch_size=batch_size)
    transitions = transitions + collector.get_transitions()
    games = games + collector.get_game_transitions()
    del collector
    collector = data_collector.DataCollector("839936889",
                                             step=steps,
                                             verbose=False,
                                             batch_size=batch_size)
    transitions = transitions + collector.get_transitions()
    games = games + collector.get_game_transitions()
    del collector
    collector = data_collector.DataCollector("837588827",
                                             step=steps,
                                             verbose=False,
                                             batch_size=batch_size)
    transitions = transitions + collector.get_transitions()
    games = games + collector.get_game_transitions()
    del collector

    t1 = time.time()

    print("data for", steps, "steps")
    print("running took", t1 - t0, "seconds")
    print("found", len(transitions), "transitions")
    print("found", len(games), "transitions")


def main():
    """

    main method

    :return:
    """

    get_step_data(16)


if __name__ == "__main__":
    main()
