"""

Author: Arthur Wesley

"""

import os

from PIL import Image
import pytesseract


def read_text(image_path):
    """

    read the text from an image

    :param image_path: path to the image
    :return:
    """

    image = Image.open(image_path)

    return pytesseract.image_to_string(image)


def main():
    """



    :return:
    """

    path = "Data/YOLO/Training Data"
    files = os.listdir(path)

    for file in files:
        print(read_text(os.path.join(path,
                                     file)))


if __name__ == "__main__":
    main()
