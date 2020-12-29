"""

runs timing tests on the data collector

"""

import time

import data_collector


def count_game_transitions(games):
    """

    count the number of game transitions

    :return: number of game transitions
    """

    count = 0

    for i in range(len(games)):

        # if this image is a lobby and the last image was not a lobby or other
        if games[i][0] == "Lobby" and games[i -1][0] in ("Gameplay", "Meeting", "Over"):
            count += 1

    return count


def get_step_data(steps):
    """

    gets run data from a given number of steps

    :param steps: number of steps to take
    :return: None (prints summary)
    """

    games = []

    t0 = time.time()

    collector = data_collector.DataCollector("825004778",
                                             step=steps,
                                             verbose=False)
    games = games + collector.get_game_transitions()
    del collector
    collector = data_collector.DataCollector("846990283",
                                             step=steps,
                                             verbose=False)
    games = games + collector.get_game_transitions()
    del collector
    collector = data_collector.DataCollector("839936889",
                                             step=steps,
                                             verbose=False)
    games = games + collector.get_game_transitions()
    del collector
    collector = data_collector.DataCollector("837588827",
                                             step=steps,
                                             verbose=False)
    games = games + collector.get_game_transitions()
    del collector

    t1 = time.time()

    print("data for", steps, "steps")
    print("running took", t1 - t0, "seconds")
    print("found", len(games), "transitions")
    print("found", count_game_transitions(games), "games")


def main():
    """

    main method

    :return:
    """

    get_step_data(8)
    get_step_data(16)


if __name__ == "__main__":
    main()
