"""

author: Arthur wesley

"""

import os

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from src import constants
from src.Models.Text_Recognition import trainer, data_generator, initalizer


def take_dataset_sample(datasets,
                        subset_sizes,
                        current_fraction):
    """

    take a random sample of a dataset that is divided into subsets of different
    sizes

    :param datasets: list of sub-datasets
    :param subset_sizes: sizes of the subsets
    :param current_fraction: the current fraction of the dataset to use
    :return: random sample of sub-datasets
    """

    # get the sample sizes
    sample_size = [int(current_fraction * subset_sizes[j]) for j in range(constants.name_length)]

    current_dataset = [datasets[j].take(sample_size[j])
                       for j in range(constants.name_length)]

    # concatenate the datasets
    dataset = current_dataset[0]

    for ds in current_dataset[1:]:
        dataset = dataset.concatenate(ds)

    return dataset


def print_learning_curves(training_path,
                          test_path,
                          steps=10):
    """

    print the learning curves of a model

    :param training_path: path to the training data
    :param test_path: path to the test data
    :param steps: number of dataset steps to take
    :return: None
    """

    # initialize constants
    subset_sizes = data_generator.get_dataset_sizes(training_path)
    vocab = trainer.get_model_vocab()

    # load the data
    training_data = [data_generator.gen_dataset_batchless(training_path,
                                                          i + 1,
                                                          vocab,
                                                          None,
                                                          subset_sizes[i])
                     for i in range(constants.name_length)]
    test_data = data_generator.gen_dataset(test_path)

    print("Dataset Size", "training accuracy", "test accuracy", sep=", ")

    # take subsets from the dataset
    for i in range(steps):

        # increment i
        i += 1

        # initialize the model
        model = initalizer.init_nn(vocab)

        # take a random sample from each length of dataset
        dataset = take_dataset_sample(training_data, subset_sizes, float(i) / steps)

        # train a model on the dataset
        model.fit(dataset,
                  verbose=0,
                  epochs=300)

        training_acc = model.evaluate(dataset,
                                      verbose=0)
        test_acc = model.evaluate(test_data,
                                  verbose=0)

        print(float(i) * sum(subset_sizes) / steps, training_acc, test_acc, sep=", ")


def length_accuracy(dataset):
    """

    generates the lengthwise accuracy of a dataset

    :param dataset: dataset to consider
    :return:
    """

    model = load_model(constants.text_recognition)

    for x, y in dataset:
        # create the accuracy evaluation object
        accuracy = SparseCategoricalAccuracy()

        # make a prediction and update the state of the accuracy using it
        prediction = model.predict(x)

        accuracy.update_state(y, prediction)

        print(y[0].numpy())
        print(np.argmax(prediction, axis=-1)[0])

        print("sequences of length", x[1].shape[1] - 1,
              "had an accuracy of", accuracy.result().numpy())


def main():
    """

    main testing method

    :return:
    """

    vocab = trainer.get_model_vocab()

    training_data = data_generator.gen_dataset(os.path.join("Data",
                                                            "Meeting Identifier",
                                                            "Test Data"),
                                               vocab=vocab,
                                               shuffle=False)

    length_accuracy(training_data)

    model = load_model(constants.text_recognition)
    model.evaluate(training_data)


if __name__ == "__main__":
    main()
