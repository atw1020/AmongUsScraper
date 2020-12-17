"""

Author: Arthur wesley

"""

import cv2

from image_generator import ImageGenerator


def test():
    """

    testing method

    :return:
    """

    video_id = "829611887"

    generator = ImageGenerator(video_id)

    image = generator.next_image("Other")

    cv2.imwrite("1.png", image)
    image = generator.next_image("Other")

    cv2.imwrite("2.png", image)
    image = generator.next_image("Other")

    cv2.imwrite("3.png", image)
    image = generator.next_image("Lobby")

    cv2.imwrite("4.png", image)
    image = generator.next_image("Meeting")

    cv2.imwrite("5.png", image)
    image = generator.next_image("Meeting")

    cv2.imwrite("6.png", image)
    image = generator.next_image("Meeting")

    cv2.imwrite("7.png", image)
    image = generator.next_image("Gameplay")

    cv2.imwrite("8.png", image)
    image = generator.next_image("Gameplay")

    cv2.imwrite("9.png", image)
    image = generator.next_image("Meeting")

    cv2.imwrite("10.png", image)
    image = generator.next_image("Meeting")

    cv2.imwrite("11.png", image)
    image = generator.next_image("Gameplay")

    cv2.imwrite("12.png", image)
    image = generator.next_image("Meeting")

    cv2.imwrite("13.png", image)
    image = generator.next_image("Gameplay")

    cv2.imwrite("14.png", image)
    image = generator.next_image("Meeting")

    cv2.imwrite("15.png", image)
    image = generator.next_image("Lobby")

    cv2.imwrite("16.png", image)
    image = generator.next_image("Over")

    cv2.imwrite("17.png", image)
    image = generator.next_image("Over")

    cv2.imwrite("18.png", image)
    image = generator.next_image("Over")

    cv2.imwrite("19.png", image)
    image = generator.next_image("Over")

    cv2.imwrite("20.png", image)
    image = generator.next_image("Over")

    cv2.imwrite("21.png", image)
    image = generator.next_image("Over")

    cv2.imwrite("22.png", image)
    image = generator.next_image("Over")

    cv2.imwrite("23.png", image)
    image = generator.next_image("Over")

    cv2.imwrite("24.png", image)
    image = generator.next_image("Lobby")

    cv2.imwrite("25.png", image)
    image = generator.next_image("Gameplay")

    cv2.imwrite("26.png", image)


def main():
    """

    main testing method

    :return:
    """

    test()


if __name__ == "__main__":
    main()
