"""

Author: Arthur Wesley

"""

from tensorflow.keras import layers
from tensorflow.keras.models import Model

from src import constants


def init_hyperparameters():
    """

    initializes

    :return:
    """


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
        hp =

    num_layers = hp.Int("Convolution Layers", 5, 10)

    vertical_convolution_size = hp.Int("Convolution Layers", 5, 15)
    horizontal_convolution_size = hp.Int("Convolution Layers", 5, 15)

    output_channels = 5 + len(vocab)

    input_layer = layers.Input(shape=image_dimensions + (3,))

    previous = input_layer

    for i in range(num_layers):
        pass

    output = previous


    assert output.output_shape == constants.yolo_output_grid_dim + (output_channels,)

    model = Model(inputs=input_layer,
                  outputs=output)
