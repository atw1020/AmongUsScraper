"""

Author: Arthur wesley, Gregory Ghiroli

"""

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image

from src.python import constants

import os


def init_nn():
    """

    initializes the neural network

    :return: game classifier neural network
    """

    # input layer
    input_layer = layers.Input(shape=constants.dimensions + (3,))

    # 2D convolutions
    convolution =   layers.Conv2D(filters=8, kernel_size=11, strides=5, activation="relu", padding="same")(input_layer)
    dropout     =   layers.Dropout(rate=constants.dropout)(convolution)
    pooling     =   layers.MaxPooling2D(pool_size=2)(dropout)
    convolution2=   layers.Conv2D(filters=16, kernel_size=11, strides=5, activation="relu", padding="same")(pooling)
    dropout2    =   layers.Dropout(rate=constants.dropout)(convolution2)
    convolution3=   layers.Conv2D(filters=32, kernel_size=11, strides=5, activation="relu", padding="same")(dropout2)
    dropout3    =   layers.Dropout(rate=constants.dropout)(convolution3)

    # flatten & feed into fully connected layers
    flatten = layers.Flatten()(dropout3)
    dense = layers.Dense(units=200, activation="relu")(flatten)
    dropout4 = layers.Dropout(rate=constants.dropout)(dense)
    dense2 = layers.Dense(units=100, activation="relu")(dropout4)
    dropout5 = layers.Dropout(rate=constants.dropout)(dense2)
    dense3 = layers.Dense(units=5, activation="relu")(dropout5)
    output = layers.Softmax()(dense3)

    model = keras.Model(inputs=input_layer, outputs=output, name="Game_Classifier")
    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

    return model


def import_image(file_path):
    """

    converts an image from a file path to a numpy array

     https://machinelearningmastery.com/how-to-normalize-center-and-standardize-images-with-the-imagedatagenerator-in-keras/

    :param file_path: path to the image
    :return: numpy array representation of the image
    """

    return image.img_to_array(image.load_img(file_path, target_size=constants.dimensions))


def main():
    """

    test method

    :return:
    """

    print(os.getcwd())

    result = import_image("Design Docs/Resources/Case 2: Gameplay.jpg")
    print(result.shape)

    model = init_nn()
    model.summary()


if __name__ == "__main__":
    main()
