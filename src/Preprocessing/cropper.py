"""

Author: Arthur Wesley

"""

import os

from PIL import Image

from src import constants


def crop_end_screen(path, output_path, start_size, box):
    """

    crops an image down to just the cremates

    :param path: path to the PIL image
    :param output_path: path to the output image
    :param start_size: initial size of the image
    :param box: the box to crop the image
    :return: None (saves cropped image
    """

    im = Image.open(path)

    assert start_size == im.size

    im = im.crop(box)

    im.save(output_path)


def crop_crewmates(path, output_dir):
    """

    crop the crewmates out of the image

    :param path: path to the image
    :param output_dir: directory to output the images to
    :return: None
    """

    im = Image.open(path)

    boxes = [
        (95, 15, 150, 90),
        (140, 20, 195, 95),
        (195, 25, 250, 100),
        (260, 30, 315, 105),
        (305, 25, 360, 100),
        (365, 20, 420, 95),
        (415, 10, 470, 85),
        (460, 5, 515, 80)
    ]

    for i in range(len(boxes)):

        temp = im.crop(boxes[i])

        new_name = "Crewmate-" + str(i) + "-" + os.path.basename(path)

        temp.save(os.path.join(output_dir, new_name))


def crop_all_crewmates(directory, output_directory):
    """

    crops all of the crewmates in specified directory

    :param directory: directory to get the images from
    :param output_directory: directory to output the images to
    :return: None
    """

    files = os.listdir(directory)

    for file in files:
        crop_crewmates(os.path.join(directory, file), output_directory)


def crop_images(directory):
    """

    crops all of the images in a specified directory

    :param directory: directory to crop all the images in
    :return: None
    """

    files = os.listdir(directory)

    for file in files:

        path = os.path.join(directory, file)

        try:
            crop_end_screen(path,
                            path,
                            constants.dimensions,
                            constants.winner_identifier_cropping)
        except AssertionError:
            print("Error Image in " + file + " had incorrect dimensions")
            continue

    print("finished cropping", directory)


def main():
    """

    main method

    crops all images in the winner identifier

    :return:
    """

    crop_all_crewmates(os.path.join("Data",
                                    "Winner Identifier",
                                    "Training Data",
                                    "ext"),
                       os.path.join("Data",
                                    "Crewmate Identifier",
                                    "Training Data",
                                    "ext"))


if __name__ == "__main__":
    main()
