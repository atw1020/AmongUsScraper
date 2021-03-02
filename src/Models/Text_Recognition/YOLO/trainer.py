"""

Author: Arthur Wesley

"""

from src.Models.Text_Recognition import text_utils
from src.Models.Text_Recognition.YOLO import data_generator, initializer


def train_network(dataset,
                  vocab):
    """



    :param dataset: dataset to train on
    :param vocab: vocabulary the network uses
    :return: trained model
    """

    model = initializer.init_nn(vocab)

    model.fit(dataset,
              epochs=10)


def main():
    """

    main testing method

    :return:
    """

    vocab = text_utils.get_model_vocab()

    dataset = data_generator.gen_dataset("Data/YOLO/Training Data",
                                         vocab=vocab)

    train_network(dataset,
                  vocab)


if __name__ == "__main__":
    main()