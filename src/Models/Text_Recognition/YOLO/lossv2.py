"""

Author: Arthur Wesley

loss function based on data from:
    https://fairyonice.github.io/Part_4_Object_Detection_with_Yolo_using_VOC_2012_data_loss.html

"""

import numpy as np

import tensorflow as tf
from tensorflow import math
from tensorflow.keras.losses import Loss

from src import constants
from src.Models.Text_Recognition import text_utils
from src.Models.Text_Recognition.YOLO import data_generator


def vectorized_IoU(y_true, y_pred):
    """

    computes a vectorized version of the IoU (intersection over union)

    :param y_true: ture y
    :param y_pred: predicted y
    :return: intersection over union of all predictions
    """

    x_overlap = tf.maximum((y_true[..., 2] + y_pred[..., 2]) / 2 - tf.abs(y_true[..., 0] - y_pred[..., 0]), 0)
    y_overlap = tf.maximum((y_true[..., 3] + y_pred[..., 3]) / 2 - tf.abs(y_true[..., 1] - y_pred[..., 1]), 0)

    intersection = x_overlap * y_overlap

    true_area = y_true[..., 2] * y_true[..., 3]
    pred_area = y_pred[..., 2] * y_pred[..., 3]

    return intersection / (true_area + pred_area - intersection)


class YoloLoss(Loss):

    def __init__(self,
                 lambda_co_ord=1,
                 lambda_class=1,
                 lambda_obj=1,
                 lambda_no_obj=1,
                 anchor_boxes=constants.anchor_boxes,
                 **kwargs):
        """

        initalizes the loss

        """

        super(YoloLoss, self).__init__()

        # extract the lambdas out
        self.lambda_co_ord = lambda_co_ord
        self.lambda_class  = lambda_class
        self.lambda_obj    = lambda_obj
        self.lambda_no_obj = lambda_no_obj

        # extract the anchor boxes
        self.anchor_boxes = anchor_boxes

        # some local variables used to prevent repeat calculations

        self.l_obj = None
        self.l_no_obj = None

        self.n_obj = None

    def call(self, y_true, y_pred):
        """

        call the method

        :param y_true:
        :param y_pred:
        :return:
        """

        self.l_obj = None
        self.l_no_obj = None

        self.n_obj = None

        print("loss 1", self.loss_1(y_true, y_pred))
        print("loss 2", self.loss_2(y_true, y_pred))
        print("loss 3", self.loss_3(y_true, y_pred))

        return self.loss_1(y_true, y_pred) + self.loss_2(y_true, y_pred) + self.loss_3(y_true, y_pred)

    def loss_1(self, y_true, y_pred):
        """

        calculates the co ordinate loss

        :param y_true: true y
        :param y_pred: predicted y
        :return:
        """

        term_1 = math.square(y_true[:, :, :, :, 1:3] - y_pred[:, :, :, :, 1:3])
        term_2 = math.square(math.sqrt(y_true[:, :, :, :, 3:5]) - math.sqrt(y_pred[:, :, :, :, 3:5]))

        return self.lambda_co_ord / self.n_objects(y_true) * \
               math.reduce_sum(math.reduce_sum(term_1 + term_2, -1) * self.l_object(y_true),
                               axis=(-1, -2, -3))

    def loss_2(self, y_true, y_pred):
        """

        computes the class loss

        :param y_true: true y
        :param y_pred: predicted y
        :return:
        """

        p     = y_true[:, :, :, :, 5:]
        p_hat = y_pred[:, :, :, :, 5:]

        log_error = math.reduce_sum(p * math.log(p_hat), axis=-1)

        return - self.lambda_class / self.n_objects(y_true) * \
               math.reduce_sum(self.l_object(y_true) * log_error,
                               axis=(-1, -2, -3))

    def loss_3(self, y_true, y_pred):
        """

        computes the object loss

        :param y_true: true y
        :param y_pred: predicted y
        :return:
        """

        # get the co-ords out of the input data
        ground_truth = y_true[..., 1:5]
        predictions  = y_pred[..., 1:5]

        IoU = vectorized_IoU(ground_truth, predictions)

        term_1 = tf.reduce_sum(self.l_object(y_true) * math.square(IoU - y_pred[..., 0]),
                               axis=(-1, -2, -3)) * self.lambda_obj
        term_2 = 0

        return term_1

    def n_objects(self, y_true):
        """

        compute the number of objects in the image

        :param y_true: the true nubmer of objects in the image
        :return:
        """

        if self.n_obj is None:
            self.n_obj = tf.reduce_sum(self.l_object(y_true), axis=(-1, -2, -3))

        return self.n_obj

    def l_object(self, y_true):
        """

        compute the variable "l_obj"

        :param y_true: the true values of y
        :return:
        """

        if self.l_obj is None:

            # l_obj is just defined as p(c)
            self.l_obj = y_true[:, :, :, :, 0]

        return self.l_obj

    def l_no_object(self, y_true, y_pred):
        """

        compute the variable "l_no_obj"

        :param y_true:
        :param y_pred:
        :return:
        """

        # todo: implement

        if self.l_no_obj is None:

            # compute the highest probability

            self.l_no_obj = 0

        return self.l_noobj


def main():
    """

    main testing method

    :return:
    """

    y_pred = np.array([[[[[0.6, 0.4, 0.5, 1.1, 1.2, 0.1, 0.88]]]]])
    y_true = np.array([[[[[1,   0.5, 0.5, 1,   1.5, 0,      1]]]]])

    loss = YoloLoss()

    # 0.15693409740924835
    # 0.15693409740924835

    print(loss(y_true, y_pred))


if __name__ == "__main__":
    main()
