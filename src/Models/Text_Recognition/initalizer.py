"""

Author: Arthur Wesley

"""

import os

from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam

from src import constants
from src.Models.Text_Recognition import trainer


def init_nn(vocab):
    """

    creates the neural network

    :return: initialized model
    """

    vocab_size = len(vocab.keys())

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

    pooling = layers.MaxPooling2D(pool_size=2,
                                  strides=2)(dropout)

    convolution = layers.Conv2D(filters=32,
                                kernel_size=5,
                                strides=2,
                                activation="relu",
                                padding="same")(pooling)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(convolution)

    convolution = layers.Conv2D(filters=64,
                                kernel_size=5,
                                strides=2,
                                activation="relu",
                                padding="same")(dropout)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(convolution)

    pooling = layers.MaxPooling2D(pool_size=2,
                                  strides=2)(dropout)

    flatten = layers.Flatten()(pooling)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(flatten)

    repeat = layers.RepeatVector(constants.name_length)(dropout)

    GRU = layers.GRU(128,
                     input_shape=(None, dropout.type_spec.shape[1]),
                     activation="relu",
                     return_sequences=True)(repeat)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(GRU)

    GRU = layers.GRU(128,
                     input_shape=(None, dropout.type_spec.shape[1]),
                     activation="relu",
                     return_sequences=True)(dropout)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(GRU)

    dense = layers.Dense(units=vocab_size)(dropout)
    softmax = layers.Softmax()(dense)

    model = Model(inputs=input_layer,
                  outputs=softmax,
                  name="Text Reader")

    opt = Adam(lr=0.001)

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])

    return model


def main():
    """

    main method

    :return:
    """

    labels = trainer.get_vocab(os.path.join("Data",
                                            "Meeting namer",
                                            "Training data"))

    model = init_nn(labels)
    model.summary()
    plot_model(model, to_file="RNN.png")


if __name__ == "__main__":
    main()
