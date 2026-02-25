#!/usr/bin/env python3
"""
Module for cropping images.
"""
import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image.

    Args:
        image: A 3D tf.Tensor containing the image to crop
        size: A tuple containing the size of the crop

    Returns:
        The cropped image
    """
    return tf.image.random_crop(image, size)
