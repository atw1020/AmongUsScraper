"""

Author: Arthur Wesley

"""

import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory

from src.python import constants
from src.python.Models.Game_Classifier import trainer


def get_failed_training_images():
    """

    prints a list of all the training images that the trained model classifies incorrectly

    :return:
    """

    # load the model
    model = tf.keras.models.load_model("Game Classifier.h5")

    # load the data
    training_data = image_dataset_from_directory("Data/Game Classifier/Training Data",
                                                 image_size=constants.dimensions)
    test_data = image_dataset_from_directory("Data/Game Classifier/Test Data",
                                             image_size=constants.dimensions)

    model.evaluate(training_data)
    model.evaluate(test_data)


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


def main():
    """

    main method

    :return:
    """

    # compute_learning_curves("test")
    get_failed_training_images()


if __name__ == "__main__":
    main()
