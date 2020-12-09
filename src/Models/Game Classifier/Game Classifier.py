"""

Author: Arthur wesley, Gregory Ghiroli

"""

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image

from src import constants


def init_nn():
    """

    initializes the neural network

    :return: game classifier neural network
    """

    input_layer = layers.Input(input_shape=constants.dimensions + (3,))
    padding = layers.ZeroPadding3D()(input_layer)
    convolution = layers.Conv3D()(padding)
    flatten = layers.Flatten()(convolution)
    dense = layers.Dense()(flatten)
    output = layers.Dense()(dense)

    return keras.Model(input_layer=input_layer, outputs=output, name="Game Classifier")


def import_image(file_path):
    """

    converts an image from a file path to a numpy array

     https://machinelearningmastery.com/how-to-normalize-center-and-standardize-images-with-the-imagedatagenerator-in-keras/

    :param file_path: path to the image
    :return: numpy array representation of the image
    """

    return image.img_to_array(image.load_img(file_path, target_size=(1280, 720)))


def main():
    """

    test method

    :return:
    """

    result = import_image("Game Classifier/Case 2: Gameplay.jpg")

    print(result.shape)

    # model = init_nn()
    # model.summary()


if __name__ == "__main__":
    main()
