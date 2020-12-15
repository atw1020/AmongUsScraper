"""

Author: Arthur wesley, Gregory Ghiroli

"""

"""

Common video resolutions

"""
res_1080p = (1920, 1080)
res_720p = (1280, 720)
res_480p = (852, 480)
res_360p = (640, 360)
res_160p = (284, 160)

# dimensions of our images
dimensions = res_360p
quality = str(dimensions[1]) + "p"

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

# dropout rate
dropout = 0.2

learning_curve_extension = " test data.txt"

"""

file I/O Constants

"""

delimiter = ", "


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
