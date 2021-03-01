"""

author: Arthur wesley

"""

import os

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import SparseCategoricalAccuracy

from src import constants
from src.Models.Text_Recognition import initalizer, trainer, data_generator


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
    sample_size = [int(current_fraction * subset_sizes[j])
                   for j in range(constants.name_length)]

    # unbatch the elements
    datasets = [dataset.unbatch() for dataset in datasets]

    current_dataset = [datasets[j].take(sample_size[j])
                       for j in range(constants.name_length)]

    # batch the datasets
    for i in range(constants.name_length):

        if sample_size[i] != 0:
            current_dataset[i] = current_dataset[i].batch(sample_size[i])

    # concatenate the datasets
    i = 0

    while sample_size[i] < 1:
        # keep increasing i until we find something that works
        i += 1

    dataset = current_dataset[i]
    i += 1

    # concatenate the rest
    while i < constants.name_length:

        # concatenate only if data exists
        if sample_size[i] > 0:
            dataset = dataset.concatenate(current_dataset[i])

        # either way, increment the loop counter
        i += 1

    return dataset.shuffle(buffer_size=1000)


def print_learning_curves(training_path,
                          test_path,
                          run_name,
                          steps=10,
                          trials=5,
                          input_shape=constants.meeting_dimensions):
    """

    print the learning curves of a model

    :param training_path: path to the training data
    :param test_path: path to the test data
    :param run_name: name of the current treatment group
    :param steps: number of dataset steps to take
    :param trials: number of trials to take
    :param input_shape: shape of the input images
    :return: None
    """

    # initialize constants
    subset_sizes = data_generator.get_dataset_sizes(training_path)
    vocab = trainer.get_model_vocab()

    # load the data
    training_data = [data_generator.gen_dataset_batchless(training_path,
                                                          i + 1,
                                                          vocab,
                                                          1,
                                                          subset_sizes[i],
                                                          input_shape)
                     for i in range(constants.name_length)]

    test_data = data_generator.gen_dataset(test_path,
                                           input_dim=input_shape,
                                           vocab=vocab)

    print("treatment", "Dataset Size", "training accuracy", "test accuracy", sep=", ")

    # take subsets from the dataset
    for i in range(steps):

        # increment i
        i += 1

        # take a random sample from each length of dataset
        dataset = take_dataset_sample(training_data, subset_sizes, float(i) / steps)

        for j in range(trials):

            # initialize the model
            model = initalizer.init_nn(vocab,
                                       image_dimensions=input_shape)

            # train a model on the dataset
            model.fit(dataset,
                      verbose=0,
                      epochs=300)

            training_acc = model.evaluate(dataset,
                                          verbose=0)
            test_acc = model.evaluate(test_data,
                                      verbose=0)

            print(run_name,
                  int(float(i) * sum(subset_sizes) / steps),
                  training_acc[1],
                  test_acc[1],
                  sep=", ")


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

    reduced_test_data = data_generator.gen_dataset(os.path.join("Data",
                                                                "Meeting Identifier",
                                                                "Reduced High Res Test Data"),
                                                   vocab=vocab,
                                                   shuffle=False,
                                                   input_dim=constants.meeting_dimensions_420p)

    test_data = data_generator.gen_dataset(os.path.join("Data",
                                                        "Meeting Identifier",
                                                        "High Res Test Data"),
                                           vocab=vocab,
                                           shuffle=False,
                                           input_dim=constants.meeting_dimensions_420p)

    model = load_model(constants.text_recognition)

    model.evaluate(reduced_test_data)
    model.evaluate(test_data)

    """length_accuracy(training_data)"""

    """print_learning_curves("Data/Meeting Identifier/Reduced High Res Training Data",
                          "Data/Meeting Identifier/High Res Test Data",
                          "480p reduced model",
                          input_shape=constants.meeting_dimensions_420p,
                          trials=3)"""


if __name__ == "__main__":
    main()
