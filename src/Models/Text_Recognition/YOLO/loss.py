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
        """squared_error = tf.map_fn(lambda x: self.mappable_loss_update(x[0], x[1]),
                                  stack)"""

        """pc_loss = tf.map_fn(lambda x: self.mappable_pc_loss(x[0], x[1]),
                            stack)"""
        pc_loss = tf.map_fn(lambda x: self.mappable_log_pc_loss(x[0], x[1]),
                             stack_2)
        mse_loss = tf.map_fn(lambda x: self.mappable_mse_loss(x[0], x[1]),
                             stack)

        """tf.print(y_true)
        tf.print(pc_loss)
        tf.print(mse_loss)"""

        squared_error = pc_loss + mse_loss

        losses = tf.reduce_mean(squared_error, axis=-1)

        return losses

    def mappable_loss_update(self, squared_error, y_true):
        """

        a function that computes the correct squared errors that can be mapped

        :squared_error
        :return:
        """

        # get the number of output channels
        output_channels = y_true.shape[-1]

        # reshape y
        y_true_first_term = tf.reshape(y_true, shape=y_true.shape + (1,))

        # reshape the squared errors
        reshaped_squared_errors = tf.reshape(squared_error,
                                             shape=squared_error.shape + (1,))
        repeated_first_term = tf.repeat(reshaped_squared_errors[:, :, 0, :],
                                        output_channels, axis=-1)

        raw_squared_error = tf.multiply(squared_error, y_true_first_term[:, :, 0, :])
        first_item_squared_error = tf.multiply(repeated_first_term, (1 - y_true_first_term[:, :, 0, :]))

        return raw_squared_error + first_item_squared_error

    def mappable_pc_loss(self, squared_error, y_true):
        """

        a mappable tensorflow function that calculates the loss caused by pc (probability of seeing
        an object)

        :param squared_error: squared errors
        :param y_true: true y
        :return: pc loss
        """

        # get the number of output channels
        output_channels = y_true.shape[-1]

        # reshape y
        y_true_first_term = tf.reshape(y_true, shape=y_true.shape + (1,))

        # reshape the squared errors
        reshaped_squared_errors = tf.reshape(squared_error,
                                             shape=squared_error.shape + (1,))
        repeated_first_term = tf.repeat(reshaped_squared_errors[:, :, 0, :],
                                        output_channels, axis=-1)

        first_item_squared_error = tf.multiply(repeated_first_term, (1 - y_true_first_term[:, :, 0, :]))

        return self.pc_lambda * first_item_squared_error

    def mappable_log_pc_loss(self, y_true, y_pred):
        """

        a mappable tensorflow function that calculates the loss caused by pc (probability of seeing
        an object)

        :param y_true: true y
        :param y_pred: predicted y
        :return:
        """

        # get the number of output channels
        H, W, C = y_true.shape

        # take the first term from y_true and y_pred
        y_true_first_term = y_true[:, :, 0]
        y_pred_first_term = y_pred[:, :, 0]

        # calculate the log error
        log_error = - (self.positive_case_lambda * tf.multiply(y_true_first_term, tf.math.log(y_pred_first_term)) +
                       self.negative_case_lambda * tf.multiply((1 - y_true_first_term), tf.math.log(abs(1 - y_pred_first_term))))

        # generate a mask for the log terms
        pc_mask = tf.ones((H, W, 1))

        # reshape the first term of y true
        y_true_mask = tf.reshape(y_true_first_term, y_true_first_term.shape + (1,))
        y_true_mask = tf.repeat(1 - y_true_mask, C - 1, axis=-1)

        mask = tf.concat([pc_mask, y_true_mask], axis=-1)

        pc_loss = tf.multiply(mask, tf.reshape(log_error, shape=log_error.shape + (1,)))
        tf.print(pc_loss, summarize=7)

        return pc_loss

    def mappable_mse_loss(self, squared_error, y_true):
        """

        a mappable tensorflow function that calculates the loss caused by mse (non PC errors

        :param squared_error: squared errors
        :param y_true: true y
        :return: mse loss of the funtion
        """

        # get the y true first term
        y_true_first_term = tf.reshape(y_true, shape=y_true.shape + (1,))

        raw_squared_error = tf.multiply(squared_error, y_true_first_term[:, :, 0, :])
        tf.print(raw_squared_error, summarize=7)

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

    y_true = tf.Variable([[[[0,    1, 0.333, 4, 0,   1,   0],
                            [1,    0.666, 0, 4, 1,   0,   0]]],
                          [[[0,    1, 0.333, 4, 0,   1,   0],
                            [1,    0.666, 0, 4, 1,   0,   0]]]])
    y_pred = tf.Variable([[[[0.99, 0.5, 0.5, 0, 0.1, 0.9, 0.01],
                            [0.99, 0.5, 0.5, 0, 0.9, 0.2, 0.01]]],
                          [[[0.99, 0.5, 0.5, 0, 0.1, 0.9, 0.01],
                            [0.99, 0.5, 0.5, 0, 0.9, 0.2, 0.01]]]],
                         dtype="float32")

    loss = YoloLoss()

    result = loss.call(y_true, y_pred)
    print("loss was\n", result.numpy())
    print(result.numpy().shape)


if __name__ == "__main__":
    main()
