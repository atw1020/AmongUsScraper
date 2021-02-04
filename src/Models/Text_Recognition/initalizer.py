"""

Author: Arthur Wesley

"""

import os
import random

from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from src import constants
from src.Models.Text_Recognition import trainer


def repeat_vector(args):
    layer_to_repeat = args[0]
    sequence_layer = args[1]
    return layers.RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)


def get_random_hyperparameters():
    """

    generate a random set of hyperparameters for the neural network

    :return: random set of hyperparameters
    """

    return {
        "conv_size": random.randint(9, 18),
        "conv_stride": random.randint(3, 4),
        "pool_1": random.randint(0, 1),
        "pool_2": 0,  # random.randint(0, 1),
        "embedding_dim": 2 ** random.randint(7, 9),
        "early_merge": 0,  # random.randint(0, 1),
        "lstm_breadth": 2 ** random.randint(8, 12),
        "lstm_depth": random.randint(1, 2),
        "end_breadth": 2 ** random.randint(8, 12),
        "end_depth": random.randint(1, 5),
        "lr": 10 ** (random.random() - 3)
    }


def init_random_nn(vocab):
    """

    initializes a neural network with random hyper-parameters

    :param vocab: vocab to create the neural network for
    :return: tuple of the network and a dictionary of its hyperparameters
    """

    kwargs = get_random_hyperparameters()
    return init_nn(vocab, **kwargs), kwargs


def init_nn(vocab,
            conv_size=5,
            conv_stride=2,
            pool_1=0,
            pool_2=0,
            embedding_dim=256,
            early_merge=0,
            lstm_breadth=512,
            lstm_depth=2,
            end_breadth=512,
            end_depth=2,
            lr=0.001):
    """

    creates the neural network

    :param embedding_dim: dimension of the embedding layer
    :param early_merge: whether or not to merge the text and image networks before or
                        after the LSTM
    :param pool_2: whether or not to use the first pooling layer
    :param pool_1: whether or not to use the second pooling layer
    :param conv_stride: stride of each convolution
    :param conv_size: size of each convolution
    :param vocab: vocabulary to use
    :param lstm_breadth: number of units in the LSTM
    :param lstm_depth: number of LSTM layers
    :param end_depth: number of layers at the end
    :param end_breadth: breadth of the last layers of the network
    :param lr: learning rate of the network
    :return: initialized model
    """

    # reset the session
    K.clear_session()

    # convert booleans from integers to booleans
    pool_1 = bool(pool_1)
    pool_2 = bool(pool_2)

    early_merge = bool(early_merge)

    # get the size of the vocabulary
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

    # RNN input layer
    rnn_input = layers.Input(shape=(None,))

    embedding = layers.Embedding(input_dim=vocab_size,
                                 output_dim=embedding_dim)(rnn_input)

    if early_merge:

        flatten_size = flatten.type_spec.shape[1]

        # repeat the flatten vector
        repeat = layers.Lambda(repeat_vector,
                               output_shape=(None, flatten_size))([flatten, embedding])

        # concatenate the embedding and repeat
        concatenate = layers.Concatenate()([embedding, repeat])
        temp = concatenate

    else:
        temp = embedding

    for i in range(lstm_depth - 1):
        LSTM = layers.LSTM(lstm_breadth,
                           activation="relu",
                           return_sequences=True)(temp)
        dropout = layers.Dropout(rate=constants.text_rec_dropout)(LSTM)
        batch_norm = layers.BatchNormalization()(dropout)

        temp = batch_norm

    LSTM = layers.LSTM(lstm_breadth,
                       activation="relu",
                       return_sequences=True)(temp)

    if not early_merge:

        flatten_size = flatten.type_spec.shape[1]

        # repeat the flatten vector
        repeat = layers.Lambda(repeat_vector,
                               output_shape=(None, flatten_size))([flatten, LSTM])

        concatenate = layers.Concatenate()([LSTM, repeat])
        temp = concatenate
    else:
        temp = LSTM

    dropout = layers.Dropout(rate=constants.text_rec_dropout)(temp)
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
