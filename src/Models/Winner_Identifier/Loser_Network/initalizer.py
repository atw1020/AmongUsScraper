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
    pooling = layers.MaxPooling2D(pool_size=3,
                                  strides=3,
                                  padding="same")(input_layer)
    convolution = layers.Conv3D(filters=8,
                                kernel_size=5,
                                strides=2,
                                activation="relu",
                                padding="same")(pooling)
    dropout = layers.Dropout(rate=0.3)(convolution)
    pooling = layers.MaxPooling2D(pool_size=2,
                                  strides=2,
                                  padding="same")(dropout)
    convolution = layers.Conv3D(filters=16,
                                kernel_size=5,
                                strides=2,
                                activation="relu",
                                padding="same")(pooling)
    dropout = layers.Dropout(rate=0.3)(convolution)
    pooling = layers.MaxPooling2D(pool_size=3,
                                  strides=3,
                                  padding="same")(dropout)
    convolution = layers.Conv3D(filters=32,
                                kernel_size=5,
                                strides=2,
                                activation="relu",
                                padding="same")(pooling)
    dropout = layers.Dropout(rate=0.3)(convolution)
    pooling = layers.MaxPooling2D(pool_size=3,
                                  strides=3,
                                  padding="same")(dropout)

    # flatten & feed into fully connected layers
    flatten = layers.Flatten()(pooling)
    dropout = layers.Dropout(rate=0.4)(flatten)
    dense = layers.Dense(units=64,
                         activation="relu")(dropout)
    dropout = layers.Dropout(rate=0.3)(dense)
    dense = layers.Dense(units=64,
                         activation="relu")(dropout)
    dropout = layers.Dropout(rate=0.3)(dense)
    dense = layers.Dense(units=32,
                         activation="relu")(dropout)
    dropout = layers.Dropout(rate=0.3)(dense)
    dense = layers.Dense(units=32,
                         activation="relu")(dropout)
    dropout = layers.Dropout(rate=0.3)(dense)

    # output layer: 12 dense units, one for each color
    output = layers.Dense(units=12, activation="sigmoid")(dropout)

    opt = Adam(learning_rate=0.0001)

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

