"""

Author: Arthur Wesley

"""

from src.Data_Collection import web_scrapper


def get_timestamp(file_name, label_length):
    """

    gets the timestamp of a frame from the name of the screenshot file

    :param file_name: name of the file
    :param label_length: number of items separated by dashes in the label
    :return: tuple: video id, vod id, frame id
    """

    file_name = file_name.split("-")
    ids = file_name[label_length:]

    # remove the file extension from the last item
    ids[-1] = ids[-1].split(".")[0]

    video = ids[0]
    vod = ids[1]

    if len(ids) > 2:
        frame = ids[2]
    else:
        frame = 0

    return video, vod, frame


def main():
    """

    main method

    :return:
    """


if __name__ == "__main__":
    main()
