"""

Author: Arthur Wesley

"""

from src.python import constants

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


def init_nn():
    """

    initializes the winner identifier neural network

    :return: initialized neural network
    """

    # input layer
    input_layer = layers.Input(shape=constants.dimensions + (3,))

    # 2D convolutions
    convolution = layers.Conv2D(filters=16, kernel_size=11, strides=5, activation="relu", padding="same")(input_layer)
    dropout = layers.Dropout(rate=constants.winner_identifier_dropout)(convolution)
    # pooling     =   layers.MaxPooling2D(pool_size=2)(dropout)
    convolution2 = layers.Conv2D(filters=32, kernel_size=11, strides=5, activation="relu", padding="same")(dropout)
    dropout2 = layers.Dropout(rate=constants.winner_identifier_dropout)(convolution2)
    convolution3 = layers.Conv2D(filters=64, kernel_size=11, strides=5, activation="relu", padding="same")(dropout2)
    dropout3 = layers.Dropout(rate=constants.winner_identifier_dropout)(convolution3)

    # flatten & feed into fully connected layers
    flatten = layers.Flatten()(dropout3)
    dense = layers.Dense(units=800, activation="relu")(flatten)
    dropout4 = layers.Dropout(rate=constants.winner_identifier_dropout)(dense)
    dense2 = layers.Dense(units=600, activation="relu")(dropout4)
    dropout5 = layers.Dropout(rate=constants.winner_identifier_dropout)(dense2)
    dense3 = layers.Dense(units=400, activation="relu")(dropout5)
    dropout6 = layers.Dropout(rate=constants.winner_identifier_dropout)(dense3)
    dense4 = layers.Dense(units=200, activation="relu")(dropout6)
    dropout7 = layers.Dropout(rate=constants.winner_identifier_dropout)(dense4)
    dense5 = layers.Dense(units=100, activation="relu")(dropout7)
    dropout8 = layers.Dropout(rate=constants.winner_identifier_dropout)(dense5)

    # output layer: 12 dense units, one for each color
    output = layers.Dense(units=12, activation="sigmoid")(dropout8)

    opt = Adam(learning_rate=0.0001)

    model = keras.Model(inputs=input_layer, outputs=output, name="Game_Classifier")
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

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

