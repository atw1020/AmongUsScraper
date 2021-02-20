"""

Author Arthur wesley

"""

from tqdm import tqdm

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

            # reset the metrics
            for metric in self.model.compiled_metrics.metrics:
                metric.reset_states()

            # go thorough the dataset
            for x, y in tqdm(dataset,
                             total=self.num_batches,
                             ncols=100,
                             unit="batch",
                             desc="training"):

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

            tqdm.write("=" * 60)

            tqdm.write("epoch " + str(i))

            for metric in self.model.compiled_metrics.metrics:
                tqdm.write("training " + metric.name + " " + str(metric.result().numpy()))

                # reset the metrics
                metric.reset_states()

            # run the metrics on the test data
            for x, y in validation_data:
                y_pred = self.model(x)

                self.model.compiled_metrics.update_state(y, y_pred)

            for metric in self.model.compiled_metrics.metrics:
                tqdm.write("test " + metric.name + " " + str(metric.result().numpy()))

            for callback in callbacks:
                callback.on_epoch_end()

            tqdm.write("=" * 60)
