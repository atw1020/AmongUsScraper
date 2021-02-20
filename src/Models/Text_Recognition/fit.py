"""

Author Arthur wesley

"""

from tensorflow import GradientTape


class ModelFitter:

    def __init__(self, model):
        """

        initialize the model fitter

        :param model:
        """

        self.model = model

    def fit(self,
            dataset,
            epochs=1,
            validation_data=None,
            callbacks=[]):

        # each epoch
        for i in range(epochs):

            # go thorough the dataset
            for x, y in dataset:

                with GradientTape() as tape:

                    # make the predictions
                    y_pred = self.model(x, training=True)

                    # compute the loss
                    loss = self.model.compiled_loss(y, y_pred)

                # compute the gradients
                trainable_vars = self.model.trainable_vars
                gradients = tape.gradient(loss, trainable_vars)

                # update the weights
                self.model.optimizer.apply_gradients(zip(gradients, trainable_vars))

                # update the metrics
                self.model.metrics.compiled_metrics.update_state(y, y_pred)
