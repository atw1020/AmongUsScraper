"""

Author: Arthur Wesley

"""

import tensorflow as tf
from tensorflow import math
from tensorflow.keras.losses import Loss


class YoloLoss(Loss):

    def __init__(self,
                 positive_case_lambda=1,
                 negative_case_lambda=1,
                 mse_lambda=1,
                 **kwargs):
        """

        initalizes the loss

        """

        super(YoloLoss, self).__init__()

        self.positive_case_lambda = positive_case_lambda
        self.negative_case_lambda = negative_case_lambda
        self.mse_lambda = mse_lambda

    def call(self, y_true, y_pred):
        """

        calculate the loss function

        :param y_true: true y
        :param y_pred: predicted y
        :return:
        """

        M, H, W, C = y_true.shape

        mask = tf.concat([tf.ones((H,  W, 1)),
                          tf.zeros((H, W, C - 1))], axis=-1)

        loss = - (tf.math.multiply_no_nan(math.log(y_pred), y_true) +
                  tf.math.multiply_no_nan(math.log(1 - y_pred), 1 - y_true))

        loss = tf.math.multiply_no_nan(loss, mask)

        return tf.reduce_mean(loss, axis=-1)


def main():
    """



    :return:
    """

    y_true = tf.Variable([[[[0, 1, 0.333, 4, 0, 1, 0],
                            [1, 0.666, 0, 4, 1, 0, 0]]],
                          [[[0, 1, 0.333, 4, 0, 1, 0],
                            [1, 0.666, 0, 4, 1, 0, 0]]]])
    y_pred = tf.Variable([[[[0.99, 0.5, 0.5, 0, 0.1, 0.9, 0.01],
                            [0.99, 0.5, 0.5, 0, 0.9, 0.2, 0.01]]],
                          [[[0.99, 0.5, 0.5, 0, 0.1, 0.9, 0.01],
                            [0.99, 0.5, 0.5, 0, 0.9, 0.2, 0.01]]]])

    loss = YoloLoss()

    result = loss.call(y_true, y_pred)
    print("loss was\n", result.numpy())
    print(result.numpy().shape)


if __name__ == "__main__":
    main()
