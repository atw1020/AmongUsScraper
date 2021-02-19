"""

Author Arthur wesley

"""

from tensorflow import GradientTape


def fit(model,
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
                y_pred = model(x, training=True)

                # compute the loss
                loss = model.compiled_loss(y, y_pred)

            # compute the gradients
            trainable_vars = model.trainable_vars
            gradients = tape.gradient(loss, trainable_vars)

            # update the weights
            model.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # update the metrics
            model.metrics.compiled_metrics.update_state(y, y_pred)
