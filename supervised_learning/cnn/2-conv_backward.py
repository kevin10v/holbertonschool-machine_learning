#!/usr/bin/env python3
"""
Module for convolutional back propagation.
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network.

    Args:
        dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
            partial derivatives with respect to the unactivated output of the
            convolutional layer
            m is the number of examples
            h_new is the height of the output
            w_new is the width of the output
            c_new is the number of channels in the output
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
                the output of the previous layer
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
           kernels for the convolution
            kh is the filter height
            kw is the filter width
        b: numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
           applied to the convolution
        padding: string that is either 'same' or 'valid', indicating the type
                 of padding used
        stride: tuple of (sh, sw) containing the strides for the convolution
            sh is the stride for the height
            sw is the stride for the width

    Returns:
        The partial derivatives with respect to the previous layer (dA_prev),
        the kernels (dW), and the biases (db), respectively
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    # Calculate padding
    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    else:  # valid
        ph, pw = 0, 0

    # Pad A_prev
    A_prev_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant', constant_values=0)

    # Initialize gradients
    dA_prev = np.zeros_like(A_prev_padded)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Compute gradients
    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                # Calculate starting positions
                h_start = i * sh
                w_start = j * sw

                # Gradient with respect to A_prev
                dA_prev[:, h_start:h_start+kh, w_start:w_start+kw, :] += (
                    W[:, :, :, k] * dZ[:, i:i+1, j:j+1, k:k+1]
                )

                # Gradient with respect to W
                region = A_prev_padded[:, h_start:h_start+kh,
                                       w_start:w_start+kw, :]
                dW[:, :, :, k] += np.sum(
                    region * dZ[:, i:i+1, j:j+1, k:k+1],
                    axis=0
                )

    # Remove padding from dA_prev
    if padding == 'same':
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_prev

    return dA_prev, dW, db
