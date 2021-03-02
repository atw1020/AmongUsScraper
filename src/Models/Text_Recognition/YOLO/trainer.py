"""

Author: Arthur Wesley

"""

from src import constants
from src.Models.Text_Recognition import text_utils
from src.Models.Text_Recognition.YOLO import data_generator, initializer
from src.Models.Text_Recognition.Recurrent_Neural_Network.trainer import TrueAccuracyCallback


def train_network(dataset,
                  vocab):
    """



    :param dataset: dataset to train on
    :param vocab: vocabulary the network uses
    :return: trained model
    """

    model = initializer.init_nn(vocab)

    cb = TrueAccuracyCallback(dataset)

    model.fit(dataset,
              epochs=50,
              callbacks=[cb])

    return model


def main():
    """

    main testing method

    :return:
    """

    vocab = text_utils.get_model_vocab()

    dataset = data_generator.gen_dataset("Data/YOLO/Training Data",
                                         vocab=vocab,
                                         batch_size=1)

    model = train_network(dataset,
                          vocab)

    model.save(constants.letter_detection)


if __name__ == "__main__":
    main()
