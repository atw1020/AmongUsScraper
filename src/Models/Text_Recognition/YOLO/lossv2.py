"""

Author: Arthur Wesley

"""

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

    def call(self, y_true, y_pred):
        """

        call the method

        :param y_true:
        :param y_pred:
        :return:
        """

        self.l_obj = None
        self.l_no_obj = None

        return self.loss_1(y_true, y_pred) + self.loss_3(y_true, y_pred) + self.loss_3(y_true, y_pred)

    def loss_1(self, y_true, y_pred):
        """

        calculates the first loss term

        :param y_true:
        :param y_pred:
        :return:
        """

        return 0

    def loss_2(self, y_true, y_pred):

        return 0

    def loss_3(self, y_true, y_pred):

        return 0

    def l_object(self, y_true, y_pred):
        """

        compute the variable "l_obj"

        :param y_true:
        :param y_pred:
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

        if self.l_no_obj is None:
            self.l_no_obj = 0

        return self.l_noobj
