"""

Author: Arthur wesley, Gregory Ghiroli

"""

from tensorflow import keras
from tensorflow.keras import layers


def init_nn():
    """

    initializes the neural network

    :return: game classifier neural network
    """

    input_layer = layers.Input()
    convolution = layers.Conv3D()(input_layer)
    dense = layers.Dense()(convolution)
    output = layers.Dense()(dense)

    return keras.Model(input_layer=input_layer, outputs=output, name="Game Classifier")


def main():
    """

    test method

    :return:
    """

    model = init_nn()
    model.summary()


if __name__ == "__main__":
    main()