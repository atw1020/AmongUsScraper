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

        # stack the squared error and true y to map them
        stack = tf.stack((y_true, y_pred), axis=1)

        # Update the Loss
        pc_loss = tf.map_fn(lambda x: self.mappable_log_pc_loss(x[0], x[1]),
                             stack)
        mse_loss = tf.map_fn(lambda x: self.mappable_mse_loss(x[0], x[1]),
                             stack)

        # take the mean of the mse loss
        mse_loss = tf.reduce_mean(mse_loss, axis=-1)

        return pc_loss + mse_loss

    def log_error(self, y_true, y_pred):
        """

        calculates the logrithmic error of y_true and y_pred

        :param y_true: actual y
        :param y_pred: predicted y
        :return: log error
        """

        return - (self.positive_case_lambda * tf.multiply(y_true, tf.math.log(y_pred)) +
                  self.negative_case_lambda * tf.multiply((1 - y_true), tf.math.log(abs(1 - y_pred))))

    def mappable_log_pc_loss(self, y_true, y_pred):
        """

        a mappable tensorflow function that calculates the loss caused by pc (probability of seeing
        an object)

        :param y_true: true y
        :param y_pred: predicted y
        :return:
        """

        # calculate the log error
        log_error = self.log_error(y_true, y_pred)

        return log_error[:, :, 0]

    def mappable_mse_loss(self, y_true, y_pred):
        """

        a mappable tensorflow function that calculates the loss caused by mse (non PC errors

        :param y_true: true y
        :param y_pred: predicted y
        :return: mse loss of the function
        """

        H, W, C = y_true.shape

        output_channels = C // self.anchor_boxes

        # compute the log loss
        log_error = self.log_error(y_true, y_pred)
        squared_error = math.square(y_pred - y_true)

        # mask out the log error and mse
        log_mask = tf.repeat(tf.concat([tf.ones((H, W, 3)),
                                        tf.zeros((H, W, 2)),
                                        tf.ones((H, W, output_channels - 5))], axis=-1),
                             repeats=self.anchor_boxes,
                             axis=-1)

        print(log_error)

        error = tf.multiply(squared_error, 1 - log_mask) + tf.math.multiply_no_nan(log_error, log_mask)

        # reshape y
        y_true_first_term = tf.reshape(y_true, shape=y_true.shape + (1,))

        raw_squared_error = tf.multiply(error, y_true_first_term[:, :, 0, :])
        # tf.print(raw_squared_error)

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

    y_true = tf.Variable([[[[0, 1, 0.333, 4, 0, 1, 0],
                            [1, 0.666, 0, 4, 1, 0, 0]]],
                          [[[0, 1, 0.333, 4, 0, 1, 0],
                            [1, 0.666, 0, 4, 1, 0, 0]]]])
    y_pred = tf.Variable([[[[0.99, 0.5, 0.5, 0, 0.1, 0.9, 0.01],
                            [0.99, 0.5, 0.5, 0, 0.9, 0.2, 0.01]]],
                          [[[0.99, 0.5, 0.5, 0, 0.1, 0.9, 0.01],
                            [0.99, 0.5, 0.5, 0, 0.9, 0.2, 0.01]]]])

    loss = YoloLoss(anchor_boxes=1)

    result = loss.call(y_true, y_pred)
    print(result)
    print("loss was", result.numpy())
    print(result.numpy().shape)


if __name__ == "__main__":
    main()
