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

        # multiply the non pc (probability that an object exists) parameters by the whether or not
        # an image was contained

        squared_error = np.array([[squared_error[i][0] if y_true[i][0] == 0
                                  else squared_error[i][j]
                                  for j in range(len(squared_error[i]))]
                                  for i in range(len(squared_error))])

        return tf.reduce_mean(squared_error, axis=-1)


def main():
    """



    :return:
    """

    y_true = np.array([[0, 2, 3, 4],
                       [1, 2, 3, 4]])
    y_pred = np.array([[1, 0, 0, 0],
                       [1, 0, 0, 0]],
                      dtype="float64")

    loss = YoloLoss()

    result = loss.call(y_true, y_pred)
    print("loss was", result.numpy())


if __name__ == "__main__":
    main()
