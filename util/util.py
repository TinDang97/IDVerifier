import cv2
import numpy


def normalize_L2(x):
    return x / numpy.sqrt((x**2).sum(axis=1, keepdims=True))


def cosine_similarity(a, b):
    assert isinstance(a, numpy.ndarray) and isinstance(b, numpy.ndarray)
    assert a.shape == b.shape

    if len(a.shape) == 1:
        return (a @ b.T) / (numpy.linalg.norm(a)*numpy.linalg.norm(b))

    return (a @ b.T) / (normalize_L2(a)*normalize_L2(b))


def resize_image(image=None, size=None):
    """
    :param image: numpy.ndarray
    :param size: (width, height)
    :return: new image
    """
    assert type(size[0]) is int and type(size[1]) is int, "Size must be int"
    assert type(image) is numpy.ndarray,\
        "Type of face features must been {}. But got {}".format(numpy.ndarray, type(image))
    _, new_height = size
    h, w, _ = image.shape

    if h == new_height:
        return image

    scale = new_height / h
    dim = int(w * scale), new_height

    return cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)