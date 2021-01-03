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
    input_layer = layers.Input(shape=constants.dimensions + (3,))

    # 2D convolutions
    convolution = layers.Conv2D(filters=8, kernel_size=3, strides=2, activation="relu", padding="same")(input_layer)
    dropout = layers.Dropout(rate=constants.winner_identifier_dropout)(convolution)
    pooling = layers.MaxPooling2D(pool_size=2, strides=2, padding="same")(dropout)
    convolution2 = layers.Conv2D(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")(pooling)
    dropout2 = layers.Dropout(rate=constants.winner_identifier_dropout)(convolution2)
    pooling2 = layers.MaxPooling2D(pool_size=2, strides=2, padding="same")(dropout2)
    convolution3 = layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(pooling2)
    dropout3 = layers.Dropout(rate=constants.winner_identifier_dropout)(convolution3)
    pooling3 = layers.MaxPooling2D(pool_size=2, strides=2, padding="same")(dropout3)

    # flatten & feed into fully connected layers
    flatten = layers.Flatten()(pooling3)
    dense = layers.Dense(units=1000, activation="relu")(flatten)
    dense2 = layers.Dense(units=200, activation="relu")(dense)
    dropout4 = layers.Dropout(rate=constants.winner_identifier_dropout)(dense2)
    dense3 = layers.Dense(units=100, activation="relu")(dropout4)
    dropout5 = layers.Dropout(rate=constants.winner_identifier_dropout)(dense3)

    # output layer: 12 dense units, one for each color
    output = layers.Dense(units=12, activation="sigmoid")(dropout5)

    opt = Adam(learning_rate=0.003)

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

