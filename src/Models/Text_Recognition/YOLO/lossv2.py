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

    def call(self, y_true, y_pred):
        """

        call the method

        :param y_true:
        :param y_pred:
        :return:
        """

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

    def loss_2(self, y_true, y_pred):

        return 0

    def l_object(self):

        return 0
