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


def init_nn(vocab,
            conv_size=5,
            conv_stride=2,
            pool_1=False,
            pool_2=False,
            lstm_breadth=512,
            end_depth=2,
            end_breadth=512,
            lr=0.001):
    """

    creates the neural network

    :param pool_2: whether or not to use the first pooling layer
    :param pool_1: whether or not to use the second pooling layer
    :param conv_stride: stride of each convolution
    :param conv_size: size of each convolution
    :param vocab: vocabulary to use
    :param lstm_breadth: number of units in the LSTM
    :param end_depth: number of layers at the end
    :param end_breadth: breadth of the last layers of the network
    :param lr: learning rate of the network
    :return: initialized model
    """

    # plus two for the start and end tokens
    vocab_size = len(vocab.keys()) + 2

    # CNN

    image_input_layer = layers.Input(shape=constants.meeting_dimensions + (3,))
    batch_norm = layers.BatchNormalization()(image_input_layer)

    convolution = layers.Conv2D(filters=16,
                                kernel_size=conv_size,
                                strides=conv_stride,
                                activation="relu",
                                padding="same")(batch_norm)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(convolution)
    batch_norm = layers.BatchNormalization()(dropout)

    convolution = layers.Conv2D(filters=32,
                                kernel_size=conv_size,
                                strides=conv_stride,
                                activation="relu",
                                padding="same")(batch_norm)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(convolution)
    batch_norm = layers.BatchNormalization()(dropout)

    temp = batch_norm

    if pool_1:
        temp = layers.MaxPooling2D(pool_size=2,
                                   strides=2)(temp)

    convolution = layers.Conv2D(filters=64,
                                kernel_size=conv_size,
                                strides=conv_stride,
                                activation="relu",
                                padding="same")(temp)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(convolution)
    batch_norm = layers.BatchNormalization()(dropout)

    convolution = layers.Conv2D(filters=128,
                                kernel_size=conv_size,
                                strides=conv_stride,
                                activation="relu",
                                padding="same")(batch_norm)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(convolution)
    batch_norm = layers.BatchNormalization()(dropout)

    temp = batch_norm

    if pool_2:
        temp = layers.MaxPooling2D(pool_size=2,
                                   strides=2)(temp)

    flatten = layers.Flatten()(temp)

    dense = layers.Dense(units=lstm_breadth,
                         activation="relu")(flatten)

    # RNN input layer
    rnn_input = layers.Input(shape=(None,))

    embedding = layers.Embedding(input_dim=vocab_size,
                                 output_dim=256)(rnn_input)

    LSTM = layers.LSTM(lstm_breadth,
                       activation="relu",
                       return_sequences=True)(embedding)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(LSTM)
    batch_norm = layers.BatchNormalization()(dropout)

    LSTM = layers.LSTM(lstm_breadth,
                       activation="relu",
                       return_sequences=False)(batch_norm)

    add = layers.Add()([LSTM, dense])
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(add)
    batch_norm = layers.BatchNormalization()(dropout)

    for i in range(end_depth):
        dense = layers.Dense(units=end_breadth,
                             activation="relu")(batch_norm)
        dropout = layers.Dropout(rate=constants.text_rec_dropout)(dense)
        batch_norm = layers.BatchNormalization()(dropout)

    dense = layers.Dense(units=vocab_size,
                         activation="sigmoid")(batch_norm)

    output = layers.Softmax()(dense)

    model = Model(inputs=[image_input_layer, rnn_input],
                  outputs=output,
                  name="Text_Reader")

    opt = Adam(lr=lr)

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])

    return model


def main():
    """

    main method

    :return:
    """

    vocab = trainer.get_vocab(os.path.join("Data",
                                            "Meeting Identifier",
                                            "Training data"))

    model = init_nn(vocab)
    model.summary()
    # plot_model(model, to_file="RNN.png")


if __name__ == "__main__":
    main()
