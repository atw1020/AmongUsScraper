"""

Author: Arthur Wesley

"""

import os

from tensorflow.keras import Model
from tensorflow.keras import layers

from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K

from kerastuner import HyperParameters

from src import constants
from src.Models.Text_Recognition import trainer


def repeat_vector(args):
    layer_to_repeat = args[0]
    sequence_layer = args[1]
    return layers.RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)


def init_hyperparameters():
    """

    initializes the hyperparameters for the model

    :return:
    """

    hp = HyperParameters()
    hp.Fixed("embedding dim", 9)

    hp.Fixed("conv_1 size", 18)
    hp.Fixed("conv_1 stride", 2)

    hp.Fixed("conv_2 size", 18)
    hp.Fixed("conv_2 stride", 2)

    hp.Fixed("conv_3 size", 18)
    hp.Fixed("conv_3 stride", 2)

    hp.Fixed("conv_4 size", 18)
    hp.Fixed("conv_4 stride", 2)

    hp.Fixed("lstm depth", 2)
    hp.Fixed("lstm breadth",  8)

    hp.Fixed("end depth",  1)
    hp.Fixed("end breadth",  8)

    hp.Fixed("learning rate", -3)
    hp.Fixed("dropout", constants.text_rec_dropout)

    return hp


def init_nn(vocab,
            hp=None,
            image_dimensions=constants.meeting_dimensions):
    """

    creates the neural network

    :param vocab: vocabulary to use
    :param hp: hyperparameters used
    :param image_dimensions: dimensions of the input images
    :return: initialized model
    """

    if hp is None:
        hp = init_hyperparameters()

    embedding_dim = int(2 ** hp.Float("embedding dim", 7, 9))

    conv_1_size = hp.Int("conv_1 size", 9, 18)
    conv_1_stride = hp.Int("conv_1 stride", 2, 4)

    conv_2_size = hp.Int("conv_2 size", 9, 18)
    conv_2_stride = hp.Int("conv_2 stride", 2, 4)

    conv_3_size = hp.Int("conv_3 size", 9, 18)
    conv_3_stride = hp.Int("conv_3 stride", 2, 4)

    conv_4_size = hp.Int("conv_4 size", 9, 18)
    conv_4_stride = hp.Int("conv_4 stride", 2, 4)

    lstm_depth = hp.Int("lstm depth", 1, 2)
    lstm_breadth = int(2 ** hp.Float("lstm breadth", 8, 12))

    end_depth = hp.Int("end depth", 1, 10)
    end_breadth = int(2 ** hp.Float("end breadth", 8, 12))
    lr = 10 ** hp.Float("learning rate", -4, -1)

    dropout_rate = hp.Float("dropout", 0.1, 0.5)

    # due to bug with tensorflow macos
    apple_silicon = False  # True

    # reset the session
    K.clear_session()

    # get the size of the vocabulary
    vocab_size = len(vocab.keys()) + 2

    # CNN

    image_input_layer = layers.Input(shape=image_dimensions + (3,))
    batch_norm = layers.BatchNormalization()(image_input_layer)

    convolution = layers.Conv2D(filters=16,
                                kernel_size=conv_1_size,
                                strides=conv_1_stride,
                                activation="relu",
                                padding="same")(batch_norm)
    dropout = layers.Dropout(rate=dropout_rate)(convolution)
    batch_norm = layers.BatchNormalization()(dropout)

    convolution = layers.Conv2D(filters=32,
                                kernel_size=conv_2_size,
                                strides=conv_2_stride,
                                activation="relu",
                                padding="same")(batch_norm)
    dropout = layers.Dropout(rate=dropout_rate)(convolution)
    batch_norm = layers.BatchNormalization()(dropout)

    temp = batch_norm

    convolution = layers.Conv2D(filters=64,
                                kernel_size=conv_3_size,
                                strides=conv_3_stride,
                                activation="relu",
                                padding="same")(temp)
    dropout = layers.Dropout(rate=dropout_rate)(convolution)
    batch_norm = layers.BatchNormalization()(dropout)

    convolution = layers.Conv2D(filters=128,
                                kernel_size=conv_4_size,
                                strides=conv_4_stride,
                                activation="relu",
                                padding="same")(batch_norm)
    dropout = layers.Dropout(rate=dropout_rate)(convolution)
    batch_norm = layers.BatchNormalization()(dropout)

    temp = batch_norm

    flatten = layers.Flatten()(temp)

    # RNN input layer
    rnn_input = layers.Input(shape=(None,))

    embedding = layers.Embedding(input_dim=vocab_size,
                                 output_dim=embedding_dim)(rnn_input)

    temp = embedding

    dense = layers.Dense(lstm_breadth,
                         activation="relu",)(flatten)

    for i in range(lstm_depth - 1):

        if apple_silicon:

            GRU = layers.GRU(lstm_breadth,
                             # recurrent_dropout=dropout_rate,
                             return_sequences=True)(temp,
                                                    initial_state=dense)

        else:

            GRU = layers.GRU(lstm_breadth,
                             recurrent_dropout=dropout_rate,
                             return_sequences=True)(temp,
                                                    initial_state=dense)

        dropout = layers.Dropout(rate=dropout_rate)(GRU)
        batch_norm = layers.BatchNormalization()(dropout)

        temp = batch_norm

    if apple_silicon:

        GRU = layers.GRU(lstm_breadth,
                         # recurrent_dropout=dropout_rate,
                         return_sequences=True)(temp,
                                                initial_state=dense)

    else:

        GRU = layers.GRU(lstm_breadth,
                         recurrent_dropout=dropout_rate,
                         return_sequences=True)(temp,
                                                initial_state=dense)

    temp = GRU

    dropout = layers.Dropout(rate=dropout_rate)(temp)
    batch_norm = layers.BatchNormalization()(dropout)

    for i in range(end_depth):
        dense = layers.Dense(units=end_breadth,
                             activation="relu")(batch_norm)
        dropout = layers.Dropout(rate=dropout_rate)(dense)
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
