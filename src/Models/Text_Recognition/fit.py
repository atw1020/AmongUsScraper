"""

Author Arthur wesley

"""

from progressbar import Bar, Percentage, ProgressBar

from tensorflow import GradientTape


class ModelFitter:

    def __init__(self, model):
        """

        initialize the model fitter

        :param model:
        """

        self.model = model
        self.num_batches = None

    def fit(self,
            dataset,
            epochs=1,
            validation_data=None,
            callbacks=[]):

        if self.num_batches is None:

            self.num_batches = 0

            for item in dataset:
                self.num_batches += 1

        # each epoch
        for i in range(epochs):

            # progressbar
            bar = ProgressBar(maxval=self.num_batches,
                              widgets=[Bar('=', '[', ']'), ' ', Percentage()])

            bar.start()

            # go thorough the dataset
            for i, (x, y) in enumerate(dataset):

                with GradientTape() as tape:

                    # make the predictions
                    y_pred = self.model(x, training=True)

                    # compute the loss
                    loss = self.model.compiled_loss(y, y_pred)

                # compute the gradients
                trainable_variables = self.model.trainable_variables
                gradients = tape.gradient(loss, trainable_variables)

                # update the weights
                self.model.optimizer.apply_gradients(zip(gradients, trainable_variables))

                # update the metrics
                self.model.compiled_metrics.update_state(y, y_pred)

                bar.update(i + 1)

            print("epoch", i, self.model.metrics.result().numpy())
