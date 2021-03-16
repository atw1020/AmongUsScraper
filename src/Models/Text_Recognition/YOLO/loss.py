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

        pc_loss  = tf.reduce_mean(pc_loss,  axis=-1)
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
        :return: mse loss of the funtion
        """

        squared_error = math.square(y_true - y_pred)
        log_error = self.log_error(y_true, y_pred)

        # unpack the shape
        H, W, C = y_true.shape

        log_mask = tf.concat(tf.ones((H, W, 3)),
                             tf.zeros((H, W, 2)),
                             tf.ones((H, W, C - 5)))

        error = tf.math.multiply_no_nan(log_error, log_mask) + tf.math.multiply(squared_error, 1 - log_mask)
        print(error.shape)

        return self.mse_lambda * error

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
