"""

Author: Arthur Wesley

"""

import os

from PIL import Image

from src import constants


def crop_image(path, output_path, start_size, box):
    """

    crops an image down to just the cremates

    :param path: path to the PIL image
    :param output_path: path to the output image
    :param start_size: initial size of the image
    :param box: the box to crop the image
    :return: None (saves cropped image
    """

    im = Image.open(path)

    assert start_size == tuple(reversed(im.size))

    im = im.crop(box)

    im.save(output_path)


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
            crop_image(path,
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

    crop_images(os.path.join("Data",
                             "Winner Identifier",
                             "losing games",
                             "Training Data",
                             "ext"))

    crop_images(os.path.join("Data",
                             "Winner Identifier",
                             "losing games",
                             "Test Data",
                             "ext"))

    crop_images(os.path.join("Data",
                             "Winner Identifier",
                             "winning games",
                             "Training Data",
                             "ext"))

    crop_images(os.path.join("Data",
                             "Winner Identifier",
                             "winning games",
                             "Test Data",
                             "ext"))


if __name__ == "__main__":
    main()
