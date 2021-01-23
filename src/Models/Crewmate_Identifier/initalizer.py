"""

Author: Arthur wesley, Gregory Ghiroli

"""

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

from src import constants

import os


def init_nn():
    """

    initializes the neural network

    :return: game classifier neural network
    """

    # input layer
    input_layer = layers.Input(shape=constants.crewmate_dimensions + (3,))

    # 2D convolutions
    convolution =   layers.Conv2D(filters=32,
                                  kernel_size=1,
                                  strides=1,
                                  activation="relu",
                                  padding="valid")(input_layer)
    dropout     =   layers.Dropout(rate=constants.crewmate_identifier_dropout)(convolution)

    convolution =   layers.Conv2D(filters=32,
                                  kernel_size=1,
                                  strides=1,
                                  activation="relu",
                                  padding="valid")(dropout)
    dropout     =   layers.Dropout(rate=constants.crewmate_identifier_dropout)(convolution)

    convolution =   layers.Conv2D(filters=64,
                                  kernel_size=5,
                                  strides=1,
                                  activation="relu",
                                  padding="valid")(dropout)
    dropout     =   layers.Dropout(rate=constants.crewmate_identifier_dropout)(convolution)

    pooling = layers.MaxPooling2D(pool_size=2,
                                  strides=2)(dropout)

    convolution = layers.Conv2D(filters=128,
                                kernel_size=5,
                                strides=1,
                                activation="relu",
                                padding="valid")(pooling)
    dropout     = layers.Dropout(rate=constants.crewmate_identifier_dropout)(convolution)

    pooling = layers.MaxPooling2D(pool_size=2,
                                  strides=2)(dropout)

    convolution = layers.Conv2D(filters=256,
                                kernel_size=5,
                                strides=1,
                                activation="relu",
                                padding="valid")(pooling)
    dropout = layers.Dropout(rate=constants.crewmate_identifier_dropout)(convolution)

    pooling = layers.MaxPooling2D(pool_size=2,
                                  strides=2)(dropout)

    # flatten & feed into fully connected layers
    flatten = layers.Flatten()(pooling)
    dropout = layers.Dropout(rate=constants.crewmate_identifier_dropout)(flatten)
    dense = layers.Dense(units=512,
                         activation="relu")(dropout)
    dropout = layers.Dropout(rate=constants.crewmate_identifier_dropout)(dense)
    dense = layers.Dense(units=512,
                         activation="relu")(dropout)
    dropout = layers.Dropout(rate=constants.crewmate_identifier_dropout)(dense)
    dense = layers.Dense(units=128,
                         activation="relu")(dropout)
    dropout = layers.Dropout(rate=constants.crewmate_identifier_dropout)(dense)
    dense = layers.Dense(units=128,
                         activation="relu")(dropout)
    dropout = layers.Dropout(rate=constants.crewmate_identifier_dropout)(dense)
    dense = layers.Dense(units=13,
                         activation="relu")(dropout)
    output = layers.Softmax()(dense)

    opt = Adam(learning_rate=0.0003)

    model = keras.Model(inputs=input_layer, outputs=output, name="Game_Classifier")
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

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

    model = init_nn()
    model.summary()


if __name__ == "__main__":
    main()
