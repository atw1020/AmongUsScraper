"""

Author: Arthur Wesley

"""

import numpy as np

import tensorflow as tf
from tensorflow import math
from tensorflow.keras.losses import Loss


class YoloLoss(Loss):

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

        # stack the squared error and true y to map them
        stack = tf.stack((squared_error, y_true), axis=0)

        # Update the Loss
        squared_error = tf.map_fn(lambda x: self.mappable_loss_update(x[0], x[1]),
                                  stack)

        return tf.reduce_mean(squared_error, axis=-1)

    def mappable_loss_update(self, squared_error, y_true):
        """

        a function that computes the correct squared errors that can be mapped

        :squared_error
        :return:
        """

        # H: Height of the output space
        # W: Width of the output space
        # O: Number of outputs
        H, W, O = y_true.shape

        return np.array([[[squared_error[i][j][0] if y_true[i][j][0] == 0
                           else squared_error[i][j][k]
                           for k in range(O)]
                           for j in range(W)]
                           for i in range(H)])


def main():
    """



    :return:
    """

    y_true = np.array([[[[0, 2, 3, 4],
                         [1, 2, 3, 4]]],
                       [[[0, 2, 3, 4],
                         [1, 2, 3, 4]]]])
    y_pred = np.array([[[[1, 0, 0, 0],
                         [1, 0, 0, 0]]],
                       [[[1, 0, 0, 0],
                         [1, 0, 0, 0]]]],
                      dtype="float64")

    loss = YoloLoss()

    result = loss.call(y_true, y_pred)
    print("loss was", result.numpy())
    print(result.numpy().shape)


if __name__ == "__main__":
    main()
