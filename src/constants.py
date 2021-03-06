"""

Author: Arthur wesley, Gregory Ghiroli

"""

"""

Neural Network names

"""

game_classifier = "Models/Game Classifier.h5"
losing_winner_identifier = "Models/Winner Identifier.h5"
winning_winner_identifier = "Models/Winner Identifier.h5"
end_screen_classifier = "Models/End Screen Classifier.h5"
crewmate_identifier = "Models/Crewmate Identifier.h5"
text_recognition = "Models/Text Recognition.h5"
letter_detection = "Models/Letter Detection.h5"

"""

Common video resolutions

"""
res_1080p = (1080, 1920)
res_720p = (720, 1280)
res_480p = (480, 852)
res_360p = (360, 640)
res_160p = (160, 284)


def quality(dim):
    """

    get the quality of images of the specified dimensions

    :param dim: dimensions to get the quality of
    :return: quality string (for twitchdl)
    """

    return str(dim[0]) + "p"


# dimensions of our images
dimensions = res_360p

sampling_rate = 1

"""

Cropping

"""

winner_identifier_cropping = (40, 160, 610, 280)
winner_identifier_dimensions = (winner_identifier_cropping[3] - winner_identifier_cropping[1],
                                winner_identifier_cropping[2] - winner_identifier_cropping[0])

crewmate_dimensions = (75, 55)
meeting_dimensions = (45, 220)
meeting_dimensions_420p = (60, 290)
meeting_dimensions_720p = (90, 435)

"""

Neural Network Parameters

"""

# accuracy goal
accuracy_objective = 0.99

# learning curve parameters
test_repeats = 10
# dataset fractions
dataset_fractions = [0.1 * i for i in range(10)]

name_length = 10

# classifier_dropout rate
classifier_dropout = 0.2  # 0.25
crewmate_identifier_dropout = 0.4
text_rec_dropout = 0.1

learning_curve_extension = " test data.txt"

"""

YOLO parameters

"""


yolo_output_grid_dim = (2, 60)
image_detection_dropoff = 0.6
ideal_letter_dimensions = (30, 30)
IoU_threshold = 0.5
anchor_boxes = 3
box_color = (21, 53, 232)

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

colors_dict = {
    "": "other",
    "RD": "red",
    "BL": "blue",
    "GN": "green",
    "PK": "pink",
    "OR": "orange",
    "YL": "yellow",
    "BK": "black",
    "WT": "white",
    "PR": "purple",
    "BN": "brown",
    "CY": "cyan",
    "LM": "lime"
}

label_ids = {
    0: "Gameplay",
    1: "Lobby",
    2: "Meeting",
    3: "Other",
    4: "Over"
}

crewmate_color_ids = [
    "black",
    "blue",
    "brown",
    "cyan",
    "green",
    "lime",
    "orange",
    "other",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
]

assert len(crewmate_color_ids) == 13


"""

frame constants

"""

end_transition_step = 50
end_screen_samples = 3
frames_per_vod = 300


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
