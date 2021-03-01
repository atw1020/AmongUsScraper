"""

Author: Arthur Wesley

"""

from kerastuner import HyperParameters

from tensorflow.keras import layers
from tensorflow.keras.models import Model

from src import constants


def init_hyperparameters():
    """

    initializes

    :return:
    """

    hp = HyperParameters()

    hp.Fixed("Convolution Layers", 5)

    hp.Fixed("Vertical Convolution", 5)
    hp.Fixed("Horizontal Convolution", 5)

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

    num_layers = hp.Int("Convolution Layers", 5, 10)

    vertical_convolution_size = hp.Int("Vertical Convolution", 5, 15)
    horizontal_convolution_size = hp.Int("Horizontal Convolution", 5, 15)

    output_channels = 5 + len(vocab)

    input_layer = layers.Input(shape=image_dimensions + (3,))

    current = input_layer

    for i in range(num_layers):
        current = layers.Conv2D(filters=2 ** (i + 2),
                                strides=1,
                                kernel_size=(vertical_convolution_size,
                                             horizontal_convolution_size),
                                padding="valid")(current)

        if i % 5 == 4:
            current = layers.MaxPool2D(pool_size=3,
                                       strides=2)(current)

    dimensions = current.type_spec.shape

    # transition to Dense-like outputs
    pseudo_dense = layers.Conv2D(filters=200,
                                 strides=1,
                                 kernel_size=(dimensions[1] + 1 - constants.yolo_output_grid_dim[0],
                                              dimensions[2] + 1 - constants.yolo_output_grid_dim[1]),
                                 padding="valid")(current)

    pseudo_dense = layers.Conv2D(filters=100,
                                 strides=1,
                                 kernel_size=1,
                                 padding="valid")(pseudo_dense)

    pseudo_dense = layers.Conv2D(filters=output_channels,
                                 strides=1,
                                 kernel_size=1,
                                 padding="valid")(pseudo_dense)

    output = pseudo_dense

    model = Model(inputs=input_layer,
                  outputs=output)

    model.compile(optimizer="Adam",
                  loss="mse")

    return model


def main():
    """



    :return:
    """

    model = init_nn({"a": 1})
    model.summary()


if __name__ == "__main__":
    main()
