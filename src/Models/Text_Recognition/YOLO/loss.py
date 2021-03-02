"""

Author: Arthur Wesley

"""

from abc import ABC

import numpy as np

import tensorflow as tf
from tensorflow import math
from tensorflow.keras.losses import Loss


class YoloLoss(Loss, ABC):

    def call(self, y_true, y_pred):
        """

        calculate the loss function

        :param y_true: true y
        :param y_pred: predicted y
        :return:
        """

        # type casting
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        # compute mean squared error
        squared_error = math.square(y_pred - y_true)

        # M: number of training Examples
        # H: Height of the output space
        # W: Width of the output space
        # O: Number of outputs
        M, H, W, O = y_true.shape

        squared_error = np.array([[[[squared_error[i][j][k][0] if y_true[i][j][k][0] == 0
                                     else squared_error[i][j][k][l]
                                     for l in range(O)]
                                     for k in range(W)]
                                     for j in range(H)]
                                     for i in range(M)])

        return tf.reduce_mean(squared_error, axis=-1)


def main():
    """



    :return:
    """

    y_true = np.array([[[[0, 2, 3, 4],
                         [1, 2, 3, 4]]],
                       [[[0, 2, 3, 4],
                         [1, 2, 3, 4]]]])
    y_pred = np.array([[[1, 0, 0, 0],
                        [1, 0, 0, 0]],
                       [[1, 0, 0, 0],
                        [1, 0, 0, 0]]],
                      dtype="float64")

    print(y_true.shape)

    loss = YoloLoss()

    result = loss.call(y_true, y_pred)
    print("loss was", result.numpy())
    print(result.numpy().shape)


if __name__ == "__main__":
    main()
