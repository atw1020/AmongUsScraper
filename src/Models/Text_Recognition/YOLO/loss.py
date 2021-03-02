"""

Author: Arthur Wesley

"""

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
        stack = tf.stack((squared_error, y_true), axis=1)

        # Update the Loss
        squared_error = tf.map_fn(lambda x: self.mappable_loss_update(x[0], x[1]),
                                  stack)

        losses = tf.reduce_mean(squared_error, axis=-1)
        print(losses)

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


def main():
    """



    :return:
    """

    y_true = tf.Variable([[[[0, 2, 3, 4],
                         [1, 2, 3, 4]]],
                       [[[0, 2, 3, 4],
                         [1, 2, 3, 4]]]])
    y_pred = tf.Variable([[[[1, 0, 0, 0],
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
