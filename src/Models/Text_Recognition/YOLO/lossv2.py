"""

Author: Arthur Wesley

"""

import tensorflow as tf
from tensorflow import math
from tensorflow.keras.losses import Loss

from src import constants
from src.Models.Text_Recognition.YOLO import box_geometry


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
        self.lambda_co_ord=lambda_co_ord
        self.lambda_co_ord=lambda_class
        self.lambda_co_ord=lambda_obj
        self.lambda_co_ord=lambda_no_obj

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

        return self.loss_1(y_true, y_pred) + self.loss_3(y_true, y_pred) + self.loss_3(y_true, y_pred)

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
               math.reduce_sum(term_1 + term_2 + self.l_object(y_true), -1)

    def loss_2(self, y_true, y_pred):
        """

        computes the class loss

        :param y_true: true y
        :param y_pred: predicted y
        :return:
        """

        p     = y_true[:, :, :, :, 5:]
        p_hat = y_pred[:, :, :, :, 5:]

        log_error = p * math.log(p_hat)

        return self.lambda_class / self.n_obj * \
               math.reduce_sum(self.l_object(y_true) * log_error)

    def loss_3(self, y_true, y_pred):

        return 0

    def n_objects(self, y_true):
        """



        :param y_true:
        :return:
        """

        if self.n_obj is None:
            return

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
