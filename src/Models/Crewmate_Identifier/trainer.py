"""

Author: Arthur Wesley

"""

from src import constants

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image_dataset_from_directory

from src.Models.Crewmate_Identifier import initalizer


def train_model(training_data, test_data):
    """

    creates and trains a model on a limited number of training examples

    :param training_data: dataset to train on
    :param test_data: test dataset
    :return: trained model
    """

    # clear the session so that we can train more than one model
    K.clear_session()

    # initialize the model
    model = initalizer.init_nn()

    # fit the model
    model.fit(training_data,
              validation_data=test_data,
              epochs=150)

    return model


def main():
    """

    main method

    :return: None
    """

    # print(os.path.exists("Data/Game Classifier/Training Data"))

    training_data = image_dataset_from_directory("Data/Crewmate Identifier/Training Data",
                                                 image_size=constants.crewmate_dimensions)

    test_data = image_dataset_from_directory("Data/Crewmate Identifier/Test Data",
                                             image_size=constants.crewmate_dimensions)

    split_data = training_data.take(len(training_data) // 2)

    # run for 200 epochs on training and test data
    model = train_model(split_data, test_data)

    model.evaluate(training_data)
    model.evaluate(test_data)

    model = train_model(training_data, test_data)

    model.predict(training_data)
    model.predict(test_data)

    model.save(constants.crewmate_identifier)


if __name__ == "__main__":
    main()
