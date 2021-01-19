"""

Author: Arthur Wesley

"""

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory

from src import constants


def main():
    """



    :return:
    """

    model = load_model(constants.end_screen_classifier)

    # load the data from the directory

    dataset = image_dataset_from_directory("Data/Winner Identifier/Training Data",
                                           image_size=constants.dimensions,
                                           shuffle=False)

    predictions = model.predict(dataset)

    print(predictions)


if __name__ == "__main__":
    main()
