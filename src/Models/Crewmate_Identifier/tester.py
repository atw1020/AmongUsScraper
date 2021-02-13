"""

Author: Arthur Wesley

"""

import os

import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

from sklearn.metrics import classification_report

from src import constants
from src.Models.Game_Classifier import trainer


def get_failed_training_images():
    """

    prints a list of all the training images that the trained model classifies incorrectly

    :return:
    """

    path = os.path.join("Data",
                        "Crewmate Identifier",
                        "Test Data")

    # load the model
    model = tf.keras.models.load_model(constants.crewmate_identifier)

    # load the data
    training_data, files = image_dataset_from_directory(path,
                                                        image_size=constants.crewmate_dimensions,
                                                        shuffle=False,
                                                        return_filepaths=True)

    step = 32
    index = 0

    # make the game_classifier_predictions
    for X, y in training_data:

        # predict and get the softmax
        y_pred = model.predict(X)
        y_pred = np.argmax(y_pred, axis=-1)

        # convert the input tensor to a numpy array
        y = y.numpy()

        if not np.array_equal(y, y_pred):
            for i in range(len(y_pred)):
                if y_pred[i] != y[i]:
                    # move the file up one directory

                    basename = os.path.basename(files[index + i])

                    os.rename(files[index + i],
                              os.path.join(path,
                                           basename))

        index += step


def compute_learning_curves(name):
    """

    computes the learning curves of the current architecture and outputs them into a text file

    :param name: name of the learning curve
    :return: None
    """

    # initialize the training data
    training_data = image_dataset_from_directory("Data/Game Classifier/Training Data")
    test_data = image_dataset_from_directory("Data/Game Classifier/Test Data")

    # number of training examples
    N = training_data.cardinality().numpy()

    # file heading
    file = open(name + constants.learning_curve_extension, "w+")

    file.write("Data Size" + constants.delimiter +
               name + " training accuracy" + constants.delimiter +
               name + " test accuracy\n")

    # iterate over all the different dataset fractions

    for dataset_fraction in constants.dataset_fractions:

        # repeat the training the specified number of times
        sample_size = int(dataset_fraction * N)
        sample = training_data.take(sample_size)

        print(type(training_data))
        print(type(sample))

        for i in range(constants.test_repeats):

            model = trainer.train_model(sample)

            training_acc = model.evaluate(sample, metrics=["acc"])
            test_acc = model.evaluate(test_data, metrics=["acc"])

            file.write(str(sample_size) + ", " + str(training_acc) + ", "+ str(test_acc) + "\n")

    file.close()


def get_training_and_test_accuracy():
    """

    gets the training and test accuracy of a model

    """

    model = tf.keras.models.load_model(constants.crewmate_identifier)

    training_data = image_dataset_from_directory("Data/Crewmate Identifier/Training Data",
                                                 image_size=constants.crewmate_dimensions)
    test_data = image_dataset_from_directory("Data/Crewmate Identifier/Test Data",
                                             image_size=constants.crewmate_dimensions)

    model.evaluate(training_data)
    model.evaluate(test_data)


def main():
    """

    main method

    :return:
    """

    # compute_learning_curves("test")
    # get_failed_training_images()

    # get_training_and_test_accuracy()

    test_data = image_dataset_from_directory("Data/Crewmate Identifier/Test Data",
                                             shuffle=False,
                                             image_size=constants.crewmate_dimensions)

    model = tf.keras.models.load_model(constants.crewmate_identifier)

    labels = np.concatenate([labels for images, labels in test_data])
    predictions = np.argmax(model.predict(test_data), axis=1)

    print(classification_report(predictions,
                                labels,
                                target_names=constants.crewmate_color_ids))


if __name__ == "__main__":
    main()
