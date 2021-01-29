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

    convolution = layers.Conv2D(filters=16,
                                kernel_size=5,
                                strides=2,
                                activation="relu",
                                padding="same")(input_layer)
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
    batch_norm = layers.BatchNormalization()(dropout)

    repeat = layers.RepeatVector(constants.name_length)(batch_norm)

    GRU = layers.SimpleRNN(256,
                           input_shape=(None, dropout.type_spec.shape[1]),
                           activation="relu",
                           return_sequences=True)(repeat)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(GRU)
    batch_norm = layers.BatchNormalization()(dropout)

    GRU = layers.SimpleRNN(128,
                           input_shape=(None, dropout.type_spec.shape[1]),
                           activation="relu",
                           return_sequences=True)(batch_norm)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(GRU)
    batch_norm = layers.BatchNormalization()(dropout)

    dense = layers.Dense(units=vocab_size,
                         activation="sigmoid")(batch_norm)

    model = Model(inputs=input_layer,
                  outputs=dense,
                  name="Text_Reader")

    opt = Adam(lr=0.01)

    model.compile(loss="binary_crossentropy",
                  optimizer=opt,
                  metrics=["binary_accuracy"])

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
