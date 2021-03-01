"""

Author: Arthur Wesley

"""

from tensorflow.keras import layers
from tensorflow.keras.models import Model

from src import constants


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

    output_channels = 5 + len(vocab)

    input_layer = layers.Input(shape=image_dimensions + (3,))

    output_layer = layers.Conv2D(filters=output_channels,
                                 kernel_size=())

    assert output_layer.output_shape == constants.yolo_output_grid_dim + (output_channels,)

    model = Model(inputs=input_layer,
                  outputs=output_layer)
