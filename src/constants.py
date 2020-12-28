"""

Author: Arthur wesley, Gregory Ghiroli

"""

"""

Neural Network names

"""

game_classifier = "Game Classifier.h5"
winner_identifier = "Winner Identifier.h5"

"""

Common video resolutions

"""
res_1080p = (1080, 1920)
res_720p = (720, 1280)
res_480p = (480, 852)
res_360p = (360, 640)
res_160p = (160, 284)

# dimensions of our images
dimensions = res_360p
quality = str(dimensions[0]) + "p"

sampling_rate = 1

"""

Neural Network Parameters

"""

# accuracy goal
accuracy_objective = 0.99

# learning curve parameters
test_repeats = 10
# dataset fractions
dataset_fractions = [0.1 * i for i in range(10)]

# classifier_dropout rate
classifier_dropout = 0.3  # 0.25
winner_identifier_dropout = 0.2

learning_curve_extension = " test data.txt"

"""

file I/O Constants

"""

delimiter = ", "

color_codes = {
    "RD": 0,
    "BL": 1,
    "GN": 2,
    "PK": 3,
    "OR": 4,
    "YL": 5,
    "BK": 6,
    "WT": 7,
    "PR": 8,
    "BN": 9,
    "CY": 10,
    "LM": 11
}

label_ids = {
    0: "Gameplay",
    1: "Lobby",
    2: "Meeting",
    3: "Other",
    4: "Over"
}


def size(res):
    """

    calculates the size in kilobytes of a color image of the given dimensions

    :param res: dimensions to calculate
    :return: size of that image
    """

    return res[0] * res[1] * 3 / 1024


def main():
    """

    main method

    :return:
    """

    print("dimensions, size (kb)")
    print("1080p", size(res_1080p), sep=", ")
    print("720p", size(res_720p), sep=", ")
    print("480p", size(res_480p), sep=", ")
    print("360p", size(res_360p), sep=", ")
    print("160p", size(res_160p), sep=", ")


if __name__ == "__main__":
    main()
