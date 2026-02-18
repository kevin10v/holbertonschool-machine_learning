#!/usr/bin/env python3
"""
Module for pooling forward propagation.
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
                the output of the previous layer
            m is the number of examples
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
        kernel_shape: tuple of (kh, kw) containing the size of the kernel for
                      the pooling
            kh is the kernel height
            kw is the kernel width
        stride: tuple of (sh, sw) containing the strides for the pooling
            sh is the stride for the height
            sw is the stride for the width
        mode: string containing either 'max' or 'avg', indicating whether to
              perform maximum or average pooling, respectively

    Returns:
        The output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    h_new = (h_prev - kh) // sh + 1
    w_new = (w_prev - kw) // sw + 1

    # Initialize output
    output = np.zeros((m, h_new, w_new, c_prev))

    # Perform pooling
    for i in range(h_new):
        for j in range(w_new):
            # Calculate starting positions
            h_start = i * sh
            w_start = j * sw
            # Extract the region of interest
            region = A_prev[:, h_start:h_start+kh, w_start:w_start+kw, :]

            # Apply pooling operation
            if mode == 'max':
                output[:, i, j, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(region, axis=(1, 2))

    return output
