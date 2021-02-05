"""

Author: Arthur Wesley

"""

import os

from twitchdl import twitch

from tensorflow.keras.preprocessing.image import save_img

from src import constants
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

    return video, int(vod), int(frame)


def update_resolution(input_dir,
                      output_dir,
                      label_length,
                      new_resolution):
    """

    convert each file in the input directory into a higher resolution copy in the
    output directory

    :param input_dir: input directory
    :param output_dir: output directory
    :param label_length: number of items separated by dashes in the label
    :param new_resolution: quality of the new images
    :return: None
    """

    files = os.listdir(input_dir)

    # sort the files by video_id ID
    files.sort(key=lambda f: get_timestamp(f, label_length)[0])

    previous_video = ""
    vods = None
    url = ""

    for file in files:

        # collect the data from the file
        video_id, vod, frame = get_timestamp(file, label_length)

        try:
            if video_id != previous_video:

                previous_video = video_id

                # update the access token
                access_token = twitch.get_access_token(video_id)
                vods = web_scrapper.get_vods(video_id, access_token)

                # get the url
                url = web_scrapper.get_base_url(video_id,
                                                access_token,
                                                constants.quality(new_resolution))
        except TypeError:
            continue

        image = web_scrapper.get_still_frame(url + vods[vod],
                                             index=frame)

        # save the image
        save_img(os.path.join(output_dir, "-".join(["Meeting", video_id, str(vod), str(frame)]) + ".jpg"),
                 image)


def main():
    """

    main method

    :return:
    """

    update_resolution("Data/Meeting Identifier/Training Data",
                      "Data/Meeting Identifier/High Res Training Data",
                      3,
                      constants.res_480p)


if __name__ == "__main__":
    main()
