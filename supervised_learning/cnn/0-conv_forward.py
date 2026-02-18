#!/usr/bin/env python3
"""
Module for convolutional forward propagation.
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural
    network.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
                the output of the previous layer
            m is the number of examples
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
           kernels for the convolution
            kh is the filter height
            kw is the filter width
            c_prev is the number of channels in the previous layer
            c_new is the number of channels in the output
        b: numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
           applied to the convolution
        activation: activation function applied to the convolution
        padding: string that is either 'same' or 'valid', indicating the type
                 of padding used
        stride: tuple of (sh, sw) containing the strides for the convolution
            sh is the stride for the height
            sw is the stride for the width

    Returns:
        The output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # Calculate padding
    if padding == 'same':
        # Calculate padding to preserve input dimensions
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
    else:  # valid
        ph, pw = 0, 0

    # Pad the input
    padded_images = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Calculate output dimensions based on padded input
    h_new = (padded_images.shape[1] - kh) // sh + 1
    w_new = (padded_images.shape[2] - kw) // sw + 1

    # Initialize output
    output = np.zeros((m, h_new, w_new, c_new))

    # Assign to standard variable name
    A_prev_padded = padded_images

    # Perform convolution
    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                # Calculate starting positions
                h_start = i * sh
                w_start = j * sw
                # Extract the region of interest
                region = A_prev_padded[:, h_start:h_start+kh,
                                       w_start:w_start+kw, :]
                # Apply convolution: element-wise multiply and sum
                output[:, i, j, k] = np.sum(region * W[:, :, :, k],
                                            axis=(1, 2, 3))

    # Add bias and apply activation
    output = activation(output + b)

    return output
