"""

Author: Arthur Wesley

"""

from kerastuner import HyperParameters

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

from src import constants
from src.Models.Text_Recognition.YOLO.loss import YoloLoss
from src.Models.Text_Recognition.YOLO.output_activation import YoloOutput


def init_hyperparameters():
    """

    initializes

    :return:
    """

    hp = HyperParameters()

    hp.Fixed("duplicate convolutional layers", 8)
    hp.Fixed("End Layers", 20)

    hp.Fixed("Vertical Convolution", 3)
    hp.Fixed("Horizontal Convolution", 3)

    hp.Fixed("MSE Lambda", 70)
    hp.Fixed("positive case Lambda", 70)

    return hp


def init_nn(vocab,
            hp=None,
            image_dimensions=constants.meeting_dimensions_420p):
    """

    initializes a neural network

    :param vocab: vocabulary to use
    :param hp: hyperparameters object
    :param image_dimensions: dimensions of the input images
    :return:
    """

    if hp is None:
        hp = init_hyperparameters()

    # duplicates = hp.Int("Convolution Layers", 5, 15)
    end_layers = hp.Int("End Layers", 1, 10)

    vertical_convolution_size = hp.Int("Vertical Convolution", 1, 15)
    horizontal_convolution_size = hp.Int("Horizontal Convolution", 1, 15)

    output_channels = 5 + len(vocab)

    input_layer = layers.Input(shape=image_dimensions + (3,))
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(input_layer)
    activation = layers.LeakyReLU()(dropout)
    current = layers.BatchNormalization()(activation)

    # num_layers = 3
    duplicates = 3

    convolution = layers.Conv2D(filters=16,
                                strides=2,
                                kernel_size=7,
                                padding="valid")(current)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(convolution)
    activation = layers.LeakyReLU()(dropout)
    current = layers.BatchNormalization()(activation)

    convolution = layers.Conv2D(filters=32,
                                strides=2,
                                kernel_size=(vertical_convolution_size,
                                             horizontal_convolution_size),
                                padding="valid")(current)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(convolution)
    activation = layers.LeakyReLU()(dropout)
    current = layers.BatchNormalization()(activation)

    current = layers.MaxPooling2D(pool_size=(2, 1),
                                  strides=(2, 1))(current)

    """for i in range(num_layers):
        convolution = layers.Conv2D(filters=int(4 ** (i + 2.5)),
                                    strides=2,
                                    kernel_size=(vertical_convolution_size,
                                                 horizontal_convolution_size),
                                    padding="valid")(current)
        dropout = layers.Dropout(rate=constants.text_rec_dropout)(convolution)
        activation = layers.LeakyReLU()(dropout)
        current = layers.BatchNormalization()(activation)

        # 1 by 1 convolutions
        for j in range(3):
            convolution = layers.Conv2D(filters=int(4 ** (i + 2.5)),
                                        strides=1,
                                        kernel_size=1,
                                        padding="valid")(current)
            dropout = layers.Dropout(rate=constants.text_rec_dropout)(convolution)
            activation = layers.LeakyReLU()(dropout)
            current = layers.BatchNormalization()(activation)"""

    for i in range(duplicates):

        convolution = layers.Conv2D(filters=128,
                                    strides=1,
                                    kernel_size=1,
                                    padding="same")(current)
        dropout = layers.Dropout(rate=constants.text_rec_dropout)(convolution)
        activation = layers.LeakyReLU()(dropout)
        current = layers.BatchNormalization()(activation)

    convolution = layers.Conv2D(filters=256,
                                strides=2,
                                kernel_size=(vertical_convolution_size,
                                             horizontal_convolution_size),
                                padding="valid")(current)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(convolution)
    activation = layers.LeakyReLU()(dropout)
    current = layers.BatchNormalization()(activation)

    for i in range(duplicates):

        convolution = layers.Conv2D(filters=256,
                                    strides=1,
                                    kernel_size=1,
                                    padding="same")(current)
        dropout = layers.Dropout(rate=constants.text_rec_dropout)(convolution)
        activation = layers.LeakyReLU()(dropout)
        current = layers.BatchNormalization()(activation)

    # complex calculations that determine the dimension of the dense YOLO layers

    # get the dimensions of the last layer
    dimensions = current.type_spec.shape
    print(dimensions)

    # update the ideal image dimensions based on how much the image has been downscaled
    ideal_height = constants.ideal_letter_dimensions[0] * dimensions[1] / image_dimensions[0]
    ideal_width  = constants.ideal_letter_dimensions[1] * dimensions[2] / image_dimensions[1]

    # calculate the best whole number stride based on the idea dimensions
    stride_x = min(max(int((dimensions[2] - ideal_width) /
                       (constants.yolo_output_grid_dim[1] - 1)), 1), 2)

    stride_y = min(max(int((dimensions[1] - ideal_height) /
                       (constants.yolo_output_grid_dim[0] - 1)), 1), 2)

    # using the exact strides, compute the actual dimensions that need to be used
    kernel_x = dimensions[2] - (constants.yolo_output_grid_dim[1] - 1) * stride_x
    kernel_y = dimensions[1] - (constants.yolo_output_grid_dim[0] - 1) * stride_y

    """kernel_x = stride_x
    kernel_y = stride_y"""

    print("stride_x", stride_x)
    print("stride_y", stride_y)

    print("kernel_x", kernel_x)
    print("kernel_y", kernel_y)

    # transition to Dense-like outputs
    pseudo_dense = layers.Conv2D(filters=2048,
                                 kernel_size=(kernel_y, kernel_x),
                                 strides=(stride_y, stride_x),
                                 padding="valid")(current)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(pseudo_dense)
    activation = layers.LeakyReLU()(dropout)
    current = layers.BatchNormalization()(activation)

    for i in range(end_layers):
        pseudo_dense = layers.Conv2D(filters=1024,
                                     strides=1,
                                     kernel_size=1,
                                     padding="valid")(current)
        dropout = layers.Dropout(rate=constants.text_rec_dropout)(pseudo_dense)
        activation = layers.LeakyReLU()(dropout)
        current = layers.BatchNormalization()(activation)

    pseudo_dense = layers.Conv2D(filters=output_channels * constants.anchor_boxes,
                                 strides=1,
                                 kernel_size=1,
                                 padding="valid")(current)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(pseudo_dense)
    current = layers.BatchNormalization()(dropout)

    output = YoloOutput(output_channels=output_channels)(current)
    # output = current
    model = Model(inputs=input_layer,
                  outputs=output)

    lr = 0.001
    print(lr)
    optimizer = Adam(learning_rate=lr)

    loss = YoloLoss(mse_lambda=1000)
    # loss = MeanSquaredError()

    model.compile(optimizer=optimizer,
                  loss=loss)

    return model


def main():
    """



    :return:
    """

    model = init_nn({"a": 1})
    model.summary()


if __name__ == "__main__":
    main()
