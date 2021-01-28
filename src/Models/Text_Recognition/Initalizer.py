"""

Author: Arthur Wesley

"""

from tensorflow.keras import Model
from tensorflow.keras import layers

from src import constants


def init_nn():
    """

    creates the neural network

    :return: initialized model
    """

    input_layer = layers.Input(shape=constants.meeting_dimensions + (3,))

    convolution = layers.Conv2D(filters=8,
                                kernel_size=5,
                                strides=2,
                                activation="relu",
                                padding="same")(input_layer)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(convolution)

    convolution = layers.Conv2D(filters=16,
                                kernel_size=5,
                                strides=2,
                                activation="relu",
                                padding="same")(dropout)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(convolution)

    convolution = layers.Conv2D(filters=32,
                                kernel_size=5,
                                strides=2,
                                activation="relu",
                                padding="same")(dropout)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(convolution)

    convolution = layers.Conv2D(filters=64,
                                kernel_size=5,
                                strides=2,
                                activation="relu",
                                padding="same")(dropout)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(convolution)

    flatten = layers.Flatten()(dropout)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(flatten)

    reshape = layers.Reshape(input_shape=())(dropout)

    GRU = layers.GRU(512,
                     input_shape=(constants.name_length, None),
                     activation="relu",
                     return_sequences=True)(dropout)
    dropout = layers.Dropout(GRU)

    GRU = layers.GRU(512,
                     input_shape=(constants.name_length, None),
                     activation="relu",
                     return_sequences=True)(dropout)
    dropout = layers.Dropout(GRU)

    dense = layers.Dense(units=constants.vocab_size,
                         activation="sigmoid",
                         )(dropout)
    softmax = layers.Softmax(dense)

    model = Model(inputs=input_layer,
                  outputs=softmax)

    return model


def main():
    """

    main method

    :return:
    """

    model = init_nn()
    model.summary()


if __name__ == "__main__":
    main()
