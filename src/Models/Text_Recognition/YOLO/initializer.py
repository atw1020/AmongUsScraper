"""

Author: Arthur Wesley

"""

from kerastuner import HyperParameters

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src import constants
from src.Models.Text_Recognition.YOLO.loss import YoloLoss
from src.Models.Text_Recognition.YOLO.output_activation import YoloOutput


def init_hyperparameters():
    """

    initializes

    :return:
    """

    hp = HyperParameters()

    hp.Fixed("Convolution Layers", 5)
    hp.Fixed("duplicate convolutional layers", 3)
    hp.Fixed("End Layers", 5)

    hp.Fixed("Vertical Convolution", 4)
    hp.Fixed("Horizontal Convolution", 5)

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

    num_layers = hp.Int("Convolution Layers", 5, 15)
    duplicates = hp.Int("duplicate convolutional layers", 1, 5)
    end_layers = hp.Int("End Layers", 1, 10)

    vertical_convolution_size = hp.Int("Vertical Convolution", 5, 15)
    horizontal_convolution_size = hp.Int("Horizontal Convolution", 5, 15)

    mse_lambda = hp.Int("MSE Lambda", 50, 500)
    positive_case_lambda = hp.Int("positive case Lambda", 50, 100)

    output_channels = 5 + len(vocab)

    input_layer = layers.Input(shape=image_dimensions + (3,))
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(input_layer)
    activation = layers.LeakyReLU()(dropout)
    current = layers.BatchNormalization()(activation)

    for i in range(num_layers):
        convolution = layers.Conv2D(filters=int(2 ** ((i + 8) / 3)),
                                    strides=1,
                                    kernel_size=(vertical_convolution_size,
                                                 horizontal_convolution_size),
                                    padding="valid")(current)
        dropout = layers.Dropout(rate=constants.text_rec_dropout)(convolution)
        activation = layers.LeakyReLU()(dropout)
        current = layers.BatchNormalization()(activation)

        for j in range(duplicates):
            convolution = layers.Conv2D(filters=int(3 ** ((i + 5) / 2)),
                                        strides=1,
                                        kernel_size=(vertical_convolution_size,
                                                     horizontal_convolution_size),
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
    stride_x = max(int((dimensions[2] - ideal_width) /
                       (constants.yolo_output_grid_dim[1] - 1) + 0.5), 1)

    stride_y = max(int((dimensions[1] - ideal_height) /
                       (constants.yolo_output_grid_dim[0] - 1) + 0.5), 1)

    # using the exact strides, compute the actual dimensions that need to be used
    kernel_x = dimensions[2] - (constants.yolo_output_grid_dim[1] - 1) * stride_x
    kernel_y = dimensions[1] - (constants.yolo_output_grid_dim[0] - 1) * stride_y

    # transition to Dense-like outputs
    pseudo_dense = layers.Conv2D(filters=80,
                                 kernel_size=(kernel_y, kernel_x),
                                 strides=(stride_y, stride_x),
                                 padding="valid")(current)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(pseudo_dense)
    activation = layers.LeakyReLU()(dropout)
    current = layers.BatchNormalization()(activation)

    for i in range(end_layers):
        pseudo_dense = layers.Conv2D(filters=50,
                                     strides=1,
                                     kernel_size=1,
                                     padding="valid")(current)
        dropout = layers.Dropout(rate=constants.text_rec_dropout)(pseudo_dense)
        activation = layers.LeakyReLU()(dropout)
        current = layers.BatchNormalization()(activation)

    pseudo_dense = layers.Conv2D(filters=output_channels,
                                 strides=1,
                                 kernel_size=1,
                                 padding="valid")(current)
    dropout = layers.Dropout(rate=constants.text_rec_dropout)(pseudo_dense)
    current = layers.BatchNormalization()(dropout)

    output = YoloOutput()(current)

    model = Model(inputs=input_layer,
                  outputs=output)

    loss = YoloLoss(mse_lambda=mse_lambda,
                    positive_case_lambda=positive_case_lambda)
    optimizer = Adam(learning_rate=0.001)

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
