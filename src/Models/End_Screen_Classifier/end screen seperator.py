"""

Author: Arthur Wesley

"""

import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory

from src import constants


def main():
    """



    :return:
    """

    path = "Data/Winner Identifier/Test Data"

    model = load_model(constants.end_screen_classifier)

    # load the data from the directory

    dataset = image_dataset_from_directory(path,
                                           image_size=constants.dimensions,
                                           shuffle=False)

    predictions = model.predict(dataset)
    files = sorted(os.listdir(os.path.join(path, "ext")))

    # go through the predictions

    for i in range(len(files)):

        # check the prediction

        if predictions[i][0] > predictions[i][1]:
            os.rename(os.path.join(path, "ext", files[i]),
                      os.path.join(path, files[i]))


if __name__ == "__main__":
    main()
