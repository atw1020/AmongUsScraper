"""

Author: Arthur Wesley

"""

import sys
import time

import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau

from src import constants
from src.Models.Text_Recognition import text_utils
from src.Models.Text_Recognition.YOLO import data_generator, initializer, text_generator


def train_network(dataset,
                  vocab,
                  reload=False):
    """



    :param dataset: dataset to train on
    :param vocab: vocabulary the network uses
    :param reload: whether or not to use a previously trained model (as opposed to a new model)
    :return: trained model
    """

    if reload:
        model = text_generator.load()
    else:
        model = initializer.init_nn(vocab,
                                    image_dimensions=constants.meeting_dimensions_420p)

    if hasattr(model, "mse_lambda"):
        print("MSE lambda is", model.loss.mse_lambda)

    model.summary()

    callbacks = [LossBreakdownCallback(dataset),
                 NanWeightsCallback(),
                 ReduceLROnPlateau(monitor="loss",
                                   cooldown=50,
                                   patience=20,
                                   verbose=1,
                                   min_lr=0.000000001)]

    model.fit(dataset,
              epochs=1000,
              callbacks=callbacks)

    return model


class LossBreakdownCallback(Callback):

    def __init__(self, training_data):
        """

        initalize the LossBreakdown Object

        :param training_data: dataset to initalize with
        """
        super().__init__()

        self.training_data = training_data

    @tf.autograph.experimental.do_not_convert
    def on_epoch_end(self, epoch, logs=None):
        """

        print the loss breakdown at the end of each epoch

        :return:
        """

        if hasattr(self.model.loss, "loss_summary"):

            t0 = time.time()

            total_pc_loss, total_mse_loss = 0, 0

            i = 0

            for x, y_true in self.training_data:

                y_pred = self.model.predict(x)

                pc_loss, mse_loss = self.model.loss.loss_summary(y_true, y_pred)

                total_pc_loss += pc_loss
                total_mse_loss += mse_loss

                i += 1

            total_pc_loss = tf.reduce_mean(total_pc_loss).numpy()
            total_mse_loss = tf.reduce_mean(total_mse_loss).numpy()

            t1 = time.time()

            print("pc loss:", total_pc_loss)
            print("mse loss:", total_mse_loss)

            print("calculation took", t1 - t0, "seconds")


class NanWeightsCallback(Callback):

    def on_epoch_end(self, epoch, logs=None):
        """

        check to see

        :param epoch:
        :param logs:
        :return:
        """

        weights = self.model.weights

        for i, layer in enumerate(weights):

            weights_sum = np.sum(layer)

            if np.isnan(weights_sum) or np.isinf(weights_sum):
                print("=" * 50, file=sys.stderr)
                print("layer", i, "has a NaN weight", file=sys.stderr)
                print("=" * 50, file=sys.stderr)

                # stop training
                self.model.stop_training = True


def main():
    """

    main testing method

    :return:
    """

    training_path = "Data/YOLO/Training Data"
    vocab = text_utils.get_model_vocab()

    dataset = data_generator.gen_dataset(training_path,
                                         vocab=vocab,
                                         batch_size=36,
                                         shuffle=False,
                                         image_dim=constants.meeting_dimensions_420p)

    model = train_network(dataset,
                          vocab,
                          reload=False)

    model.save(constants.letter_detection)


if __name__ == "__main__":
    main()
