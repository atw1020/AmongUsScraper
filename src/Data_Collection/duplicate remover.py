"""

Removes images of the same end screen from a directory

"""

import os


def is_end_screen(filename):
    """

    determines if a given filename is an end screen or not

    there are two ways a file can be an end screen
    A) it is labeled "Over"
    B) it is labeled with color names ie. "BKBN"

    :param filename: name of the file
    :return: boolean: is the filename a screen
    """

    label = filename.split("-")[0]

    if label == "Over":
        return True

    else:
        return "RD" in label or "BL" in label or "GN" in label or \
               "PK" in label or "OR" in label or "YL" in label or \
               "BL" in label or "WT" in label or "PR" in label or \
               "BN" in label or "CY" in label or "LM" in label


def get_end_duplicates(end_screen, images):
    """

    generates a list of images of the same end screen as end_screen based on the index
    of the images

    :param end_screen: file name of the end screen (assumed to be the first image sequentially)
    :param images: list of images in the directory (sorted)
    :return: list of images of the same end screen
    """

    # remove the file extension
    filename = end_screen.split(".")[0]
    filename = filename.split("-")

    video_id = filename[1]
    vod_id = int(filename[2])

    duplicates = []

    for image in images:

        # check to see if it's from the right video
        if video_id in image:

            # add the image if it shares the vod id
            if str(vod_id) in image:
                duplicates.append(image)

            # if the image is the next image sequentially, it is also a duplicate
            if str(vod_id + 1) in image:

                # increment the video ID
                vod_id += 1
                duplicates.append(image)

    return duplicates


def remove_end_duplicates(directory):
    """

    removes all of the duplicate images from a given directory

    :param directory: directory to remove the images from
    :return: None
    """

    files = sorted(os.listdir(directory))

    processed_files = []

    for file in files:

        if file in processed_files:
            # if we've already removed, this file, skip it
            continue

        if is_end_screen(file):
            duplicates = get_end_duplicates(file, files)

            # get the middle file
            middle = duplicates[int(len(duplicates) / 2)]

            # remove all the files except for the middle duplicate
            for duplicate in duplicates:

                # remove only if the duplicate is not the middle
                if duplicate != middle:
                    try:
                        os.remove(os.path.join(directory, duplicate))
                    except FileNotFoundError:
                        print(duplicates)
                        print(middle)

            # record which files we processed
            processed_files += duplicates


def get_player_duplicates(player, images):
    """

    generate a list of all the images in images that also show the specified player

    :param player: player name, color and living status to look for
    :param images: images to look through
    :return: list of duplicated images
    """

    status = "-".join(player.split("-")[:3])

    return [image for image in images if status in image]


def remove_player_duplicates(directory):
    """

    move all of the

    :param directory:
    :return:
    """

    files = os.listdir(directory)

    i = 0

    while i < len(files):

        duplicates = get_player_duplicates(files[i], files[i:])

        for duplicate in duplicates:
            # remove the file
            os.remove(os.path.join(directory, duplicate))

            files.remove(duplicate)

        i += 1



def main():
    """

    main method

    :return: None
    """

    # remove_end_duplicates("Data/Temp Images")

    path = "Data/Temp Images"

    remove_player_duplicates(path)


if __name__ == "__main__":
    main()
