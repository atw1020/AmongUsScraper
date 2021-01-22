"""

Author: Arthur Wesley

"""

from src import constants

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


def init_nn():
    """

    initializes the winner identifier neural network

    :return: initialized neural network
    """

    # input layer
    input_layer = layers.Input(shape=constants.winner_identifier_dimensions + (3,))

    # 2D convolutions
    convolution = layers.Conv2D(filters=8,
                                kernel_size=5,
                                strides=2,
                                activation="relu",
                                padding="same")(input_layer)
    dropout = layers.Dropout(rate=constants.winner_identifier_dropout)(convolution)

    convolution = layers.Conv2D(filters=16,
                                kernel_size=5,
                                strides=2,
                                activation="relu",
                                padding="same")(dropout)
    dropout = layers.Dropout(rate=constants.winner_identifier_dropout)(convolution)

    pooling = layers.MaxPooling2D(pool_size=2,
                                  strides=2,
                                  padding="same")(dropout)

    convolution = layers.Conv2D(filters=32,
                                kernel_size=5,
                                strides=2,
                                activation="relu",
                                padding="same")(pooling)
    dropout = layers.Dropout(rate=constants.winner_identifier_dropout)(convolution)

    convolution = layers.Conv2D(filters=64,
                                kernel_size=5,
                                strides=2,
                                activation="relu",
                                padding="same")(dropout)
    dropout = layers.Dropout(rate=constants.winner_identifier_dropout)(convolution)


    pooling = layers.MaxPooling2D(pool_size=2,
                                  strides=2,
                                  padding="same")(dropout)

    # flatten & feed into fully connected layers
    flatten = layers.Flatten()(pooling)
    dropout = layers.Dropout(rate=constants.winner_identifier_dropout)(flatten)
    dense = layers.Dense(units=512,
                         activation="relu")(dropout)
    dropout = layers.Dropout(rate=constants.winner_identifier_dropout)(dense)
    dense = layers.Dense(units=512,
                         activation="relu")(dropout)
    dropout = layers.Dropout(rate=constants.winner_identifier_dropout)(dense)
    dense = layers.Dense(units=128,
                         activation="relu")(dropout)
    dropout = layers.Dropout(rate=constants.winner_identifier_dropout)(dense)
    dense = layers.Dense(units=128,
                         activation="relu")(dropout)
    dropout = layers.Dropout(rate=constants.winner_identifier_dropout)(dense)

    # output layer: 12 dense units, one for each color
    output = layers.Dense(units=12, activation="sigmoid")(dropout)

    opt = Adam(learning_rate=0.0003)

    model = keras.Model(inputs=input_layer, outputs=output, name="Game_Classifier")
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["binary_accuracy"])

    return model


def main():
    """

    main

    :return:
    """

    model = init_nn()
    model.summary()


if __name__ == "__main__":
    main()

