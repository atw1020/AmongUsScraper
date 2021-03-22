"""

Author: Arthur Wesley

"""

import numpy as np

from tensorflow import concat

from tensorflow.keras.layers import Layer
from tensorflow.keras.activations import sigmoid

from src import constants


class YoloOutput(Layer):
    """

    Applies the output YOLO activation to a vector

    """

    def __init__(self,
                 output_channels,
                 **kwargs):
        """

        initialization function (only calls super)

        :param vocab: vocabulary to use
        """
        super(YoloOutput, self).__init__()

        self.output_channels = output_channels

    def get_config(self):
        """

        get the configuration of the layer

        :return: configuration of the layer
        """

        config = super().get_config().copy()
        config.update({
            "output_channels": self.output_channels
        })

        return config

    def call(self, inputs, **kwargs):
        """

        run a vector through the implementation

        :param inputs:
        :return:
        """

        tensors_to_concat = []

        # go through all the anchor boxes
        for i in range(constants.anchor_boxes):

            # compute the base of this anchor box
            anchor_base = i * self.output_channels

            start = sigmoid(inputs[:, :, :, anchor_base:anchor_base + 3])
            middle = inputs[:, :, :, anchor_base + 3:anchor_base + 5]
            end = sigmoid(inputs[:, :, :, anchor_base + 5:anchor_base + self.output_channels])

            # apply sigmoid activation to all but the width and height (items 3 and 4)
            tensors_to_concat.append(start)
            tensors_to_concat.append(middle)
            tensors_to_concat.append(end)

        # concatenate the three
        return concat(tensors_to_concat, axis=-1)


def main():
    """



    :return:
    """

    layer = YoloOutput(6)

    test_data = np.array([[[[2, 2, 2, 3, 3, 2] * constants.anchor_boxes]]],
                         dtype="float")
    print(test_data.shape)
    print(layer(test_data))


if __name__ == "__main__":
    main()
