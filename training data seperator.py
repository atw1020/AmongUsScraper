"""

Author: Arthur Wesley

"""

import os
import hashlib

from twitchdl import twitch


def is_name_test(name):
    """

    tells us if a given name is in the training or test set

    :param name: name of the streamer
    :return: whether or not their videos are in the test set
    """

    # 70% of the images are training, so return true if the modulo
    # is in the top 30%
    return string_hash(name) % 10 >= 7


def string_hash(st):
    """

    gets an integer hash of a string

    :param st: string to get the hash of
    :return: integer hash
    """

    m = hashlib.md5(st.encode("utf-8"))

    return int(m.hexdigest(), 16)


def sort_classifier_crude_data():
    """

    sorts through "crude" data and puts it into Test data

    :return: None
    """

    start_path = "Data/Game Classifier/Crude Data/"

    files = os.listdir(start_path)

    player_names = dict()

    for file in files:

        # split the filepath on the dash
        items = file.split("-")
        image_kind = items[0]
        video_id = items[1]

        if video_id in player_names:
            # if we already have the player name cached, use it
            player_name = player_names[video_id]
        else:
            # otherwise, get the player name from the video ID
            video = twitch.get_video(video_id)
            player_name = video['channel']['display_name']

            # cache the player name
            player_names[video_id] = player_name

        if is_name_test(player_name):
            data_set = "Test Data"
        else:
            data_set = "Training Data"

        output = os.path.join("Data", "Game Classifier", data_set, image_kind, file)

        os.rename(start_path + file, output)


def sort_winner_crude_data():
    """

    sorts through the data for the winner identifier and sorts it into training and test
    data

    :return: None (renames files)
    """

    path = os.path.join("Data", "Winner Identifier")

    files = os.listdir(os.path.join(path, "Crude Data"))

    player_names = dict()

    for file in files:

        items = file.split("-")

        video_id = items[1]

        if video_id in player_names:
            # if we already have the player name cached, use it
            player_name = player_names[video_id]
        else:
            # otherwise, get the player name from the video ID
            video = twitch.get_video(video_id)
            player_name = video['channel']['display_name']

            # cache the player name
            player_names[video_id] = player_name

        if is_name_test(player_name):
            data_set = "Test Data"
        else:
            data_set = "Training Data"

        os.rename(os.path.join(path, "Crude Data", file),
                  os.path.join(path, data_set, "ext", file))


def main():
    """

    main method

    :return:
    """

    sort_classifier_crude_data()
    # sort_winner_crude_data() winner identifier crude data is not sorted by
    # streamer

    """
    streamers = ["Blunder", "shofu", "Kara", "Pokimane", "5up", "captainsparklez",
                 "showthyme", "sykkuno", "GUMMI", "hafu"]

    for streamer in streamers:
        print(streamer, "is", "in" if is_name_test(streamer) else "not in", "the test set")
    """


if __name__ == "__main__":
    main()
