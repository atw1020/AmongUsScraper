"""

Author: Arthur Wesley

"""

import tensorflow as tf
from tensorflow import math
from tensorflow.keras.losses import Loss

from src import constants


class YoloLoss(Loss):

    def __init__(self,
                 positive_case_lambda=1,
                 negative_case_lambda=1,
                 mse_lambda=1,
                 anchor_boxes=constants.anchor_boxes,
                 **kwargs):
        """

        initalizes the loss

        """

        super(YoloLoss, self).__init__()

        self.positive_case_lambda = positive_case_lambda
        self.negative_case_lambda = negative_case_lambda
        self.mse_lambda = mse_lambda

        self.anchor_boxes = anchor_boxes

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
        stack = tf.stack((squared_error, y_true), axis=1)
        stack_2 = tf.stack((y_true, y_pred), axis=1)

        pc_loss = tf.map_fn(lambda x: self.mappable_log_pc_loss(x[0], x[1]),
                             stack_2)
        mse_loss = tf.map_fn(lambda x: self.mappable_mse_loss(x[0], x[1]),
                             stack)

        squared_error = pc_loss + mse_loss

        losses = tf.reduce_mean(squared_error, axis=-1)

        return losses

    def mappable_log_pc_loss(self, y_true, y_pred):
        """

        a mappable tensorflow function that calculates the loss caused by pc (probability of seeing
        an object)

        :param y_true: true y
        :param y_pred: predicted y
        :return:
        """

        # calculate the log error
        log_error = - (self.positive_case_lambda * tf.multiply(y_true, tf.math.log(y_pred)) +
                       self.negative_case_lambda * tf.multiply((1 - y_true), tf.math.log(abs(1 - y_pred))))

        # get the number of output channels
        output_channels = y_true.shape[-1]

        # reshape y
        y_true_first_term = tf.reshape(y_true, shape=y_true.shape + (1,))

        # reshape the log errors
        reshaped_squared_errors = tf.reshape(log_error,
                                             shape=log_error.shape + (1,))
        repeated_first_term = tf.repeat(reshaped_squared_errors[:, :, 0, :],
                                        output_channels, axis=-1)

        first_item_squared_error = tf.multiply(repeated_first_term, (1 - y_true_first_term[:, :, 0, :]))

        return first_item_squared_error

    def mappable_mse_loss(self, squared_error, y_true):
        """

        a mappable tensorflow function that calculates the loss caused by mse (non PC errors

        :param squared_error: squared errors
        :param y_true: true y
        :return: mse loss of the funtion
        """

        # reshape y
        y_true_first_term = tf.reshape(y_true, shape=y_true.shape + (1,))

        raw_squared_error = tf.multiply(squared_error, y_true_first_term[:, :, 0, :])

        return self.mse_lambda * raw_squared_error

    def loss_summary(self, y_true, y_pred):
        """

        prints a summary of the loss for the specified dataset (breakdown pc loss vs mse)

        :return: None (prints results)
        """

        # type casting
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        # compute mean squared error
        squared_error = math.square(y_pred - y_true)

        # stack the squared error and true y to map them
        stack = tf.stack((squared_error, y_true), axis=1)
        stack_2 = tf.stack((y_true, y_pred), axis=1)

        # Update the Loss
        pc_loss = tf.map_fn(lambda x: self.mappable_log_pc_loss(x[0], x[1]),
                            stack_2)
        mse_loss = tf.map_fn(lambda x: self.mappable_mse_loss(x[0], x[1]),
                             stack)

        pc_loss =  tf.reduce_mean(pc_loss, axis=-1)
        mse_loss = tf.reduce_mean(mse_loss, axis=-1)

        return pc_loss, mse_loss


def main():
    """



    :return:
    """

    y_true = tf.Variable([[[[0, 2, 3, 4],
                         [1, 2, 3, 4]]],
                       [[[0, 2, 3, 4],
                         [1, 2, 3, 4]]]])
    y_pred = tf.Variable([[[[0.99, 0, 0, 0],
                         [0.99, 0, 0, 0]]],
                       [[[0.99, 0, 0, 0],
                         [0.99, 0, 0, 0]]]],
                      dtype="float64")

    loss = YoloLoss()

    result = loss.call(y_true, y_pred)
    print(result)
    print("loss was", result.numpy())
    print(result.numpy().shape)


if __name__ == "__main__":
    main()
