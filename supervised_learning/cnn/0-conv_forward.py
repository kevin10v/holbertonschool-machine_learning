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

    # Calculate output dimensions
    if padding == 'same':
        h_new = int(np.ceil(h_prev / sh))
        w_new = int(np.ceil(w_prev / sw))
    else:  # valid
        h_new = (h_prev - kh) // sh + 1
        w_new = (w_prev - kw) // sw + 1

    # Calculate padding
    if padding == 'same':
        # Calculate required padding to support all output positions
        pad_h_total = max((h_new - 1) * sh + kh - h_prev, 0)
        pad_w_total = max((w_new - 1) * sw + kw - w_prev, 0)
        # Distribute padding - put more on top/left when odd
        ph = (pad_h_total + 1) // 2
        pw = (pad_w_total + 1) // 2
        ph_extra = pad_h_total - ph
        pw_extra = pad_w_total - pw
    else:
        ph, pw = 0, 0
        ph_extra, pw_extra = 0, 0

    # Pad the input (asymmetric if needed)
    A_prev_padded = np.pad(
        A_prev,
        ((0, 0), (ph, ph_extra), (pw, pw_extra), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Initialize output
    output = np.zeros((m, h_new, w_new, c_new))

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
