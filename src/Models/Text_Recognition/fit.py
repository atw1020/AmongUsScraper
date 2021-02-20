"""

Author Arthur wesley

"""

from time import sleep

from tqdm import tqdm

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

        num_training_batches = 0
        num_test_batches = 0

        for item in dataset:
            num_training_batches += 1

        for item in validation_data:
            num_test_batches += 1

        output_length = 80

        # each epoch
        for i in range(epochs):

            # reset the metrics
            for metric in self.model.compiled_metrics.metrics:
                metric.reset_states()

            tqdm.write("=" * output_length)
            sleep(0.05)

            # go thorough the dataset
            for x, y in tqdm(dataset,
                             total=num_training_batches,
                             ncols=output_length,
                             unit="batch",
                             desc="epoch " + str(i + 1) + "/" + str(epochs)):

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

            for metric in self.model.compiled_metrics.metrics:

                initial_string = "training " + metric.name + ":"
                final_string = str(metric.result().numpy())

                middle_string = " " * (output_length - len(initial_string) - len(final_string))

                tqdm.write(initial_string + middle_string + final_string)

                # reset the metrics
                metric.reset_states()

            sleep(0.05)

            # run the metrics on the test data
            for x, y in tqdm(dataset,
                             total=num_test_batches,
                             ncols=output_length,
                             unit="batch",
                             desc="testing..."):
                y_pred = self.model(x)

                self.model.compiled_metrics.update_state(y, y_pred)

            for metric in self.model.compiled_metrics.metrics:

                initial_string = "test " + metric.name + ":"
                final_string = str(metric.result().numpy())

                middle_string = " " * (output_length - len(initial_string) - len(final_string))

                tqdm.write(initial_string + middle_string + final_string)

            for callback in callbacks:
                callback.on_epoch_end()
