"""

Author: Arthur Wesley

"""


def box_area(box):
    """

    finds the area of the box

    :param box: box to find the area of
    :return: area of the box
    """

    # decompress tuple
    x, y, w, h = box

    return w * h


def overlap_distance(x1, w1, x2, w2):
    """

    finds the distance of overlap of two 1-D regions

    :param x1: the center of the first region
    :param w1: the width of the first region
    :param x2: the center of the second region
    :param w2: the width of the second region
    :return:
    """

    return (w1 + w2) / 2 - abs(x1 - x2)


def box_intersection(box1, box2):
    """

    finds the area of intersection of two boxes

    :param box1:
    :param box2:
    :return:
    """

    # unload tuples
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    horiz_overlap = overlap_distance(x1, w1,
                                     x2, w2)
    vert_overlap = overlap_distance(y1, h1,
                                    y2, h2)

    return horiz_overlap * vert_overlap


def box_union(box1, box2, intersection=None):
    """

    finds the area of union of two boxes

    :param box1:
    :param box2:
    :param intersection: pre-computed intersection of the boxes
    :return:
    """

    if intersection is None:
        intersection = box_intersection(box1, box2)

    return box_area(box1) + box_area(box2) - intersection


def main():
    """

    Main testing method

    :return:
    """




if __name__ == "__main__":
    main()