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

    # plus two for the
    vocab_size = len(vocab.keys()) + 2

    # CNN

    image_input_layer = layers.Input(shape=constants.meeting_dimensions + (3,))

    convolution = layers.Conv2D(filters=16,
                                kernel_size=5,
                                strides=2,
                                activation="relu",
                                padding="same")(image_input_layer)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(convolution)
    batch_norm = layers.BatchNormalization()(dropout)

    convolution = layers.Conv2D(filters=32,
                                kernel_size=5,
                                strides=2,
                                activation="relu",
                                padding="same")(batch_norm)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(convolution)
    batch_norm = layers.BatchNormalization()(dropout)

    convolution = layers.Conv2D(filters=64,
                                kernel_size=5,
                                strides=2,
                                activation="relu",
                                padding="same")(batch_norm)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(convolution)
    batch_norm = layers.BatchNormalization()(dropout)

    convolution = layers.Conv2D(filters=128,
                                kernel_size=5,
                                strides=2,
                                activation="relu",
                                padding="same")(batch_norm)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(convolution)
    batch_norm = layers.BatchNormalization()(dropout)

    flatten = layers.Flatten()(batch_norm)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(flatten)
    CNN_output = layers.BatchNormalization()(dropout)

    # RNN input layer
    rnn_input = layers.Input(shape=(None, vocab_size))

    LSTM = layers.LSTM(252,
                       activation="relu",
                       return_sequences=True)(rnn_input)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(LSTM)
    batch_norm = layers.BatchNormalization()(dropout)

    LSTM = layers.LSTM(252,
                       activation="relu",
                       return_sequences=False)(batch_norm)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(LSTM)
    RNN_output = layers.BatchNormalization()(dropout)

    concatenate = layers.Concatenate()([CNN_output, RNN_output])

    dense = layers.Dense(units=256,
                         activation="sigmoid")(concatenate)

    output = layers.Dense(units=vocab_size,
                          activation="sigmoid")(dense)

    model = Model(inputs=[image_input_layer, rnn_input],
                  outputs=output,
                  name="Text_Reader")

    opt = Adam(lr=0.001)

    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["categorical_accuracy"])

    return model


def main():
    """

    main method

    :return:
    """

    labels = trainer.get_vocab(os.path.join("Data",
                                            "Meeting Identifier",
                                            "Training data"))

    model = init_nn(labels)
    model.summary()
    plot_model(model, to_file="RNN.png")


if __name__ == "__main__":
    main()
