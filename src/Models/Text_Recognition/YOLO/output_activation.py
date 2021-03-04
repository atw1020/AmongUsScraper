"""

Author: Arthur Wesley

"""

from tensorflow import concat

from tensorflow.keras.layers import Layer
from tensorflow.keras.activations import sigmoid


class YoloOutput(Layer):
    """

    Applies the output YOLO activation to a vector

    """

    def __init__(self, **kwargs):
        """

        initialization function (only calls super)

        """
        super(YoloOutput, self).__init__()

    def call(self, inputs, **kwargs):
        """

        run a vector through the implementation

        :param inputs:
        :return:
        """

        # apply sigmoid activation to all but the width and height (items 3 and 4)
        start = sigmoid(inputs[:, :, :, :3])
        middle = inputs[:, :, :, 3:5]
        end = sigmoid(inputs[:, :, :, 5:])

        # concatenate the three
        return concat([start, middle, end], axis=-1)
