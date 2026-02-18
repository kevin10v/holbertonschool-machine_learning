#!/usr/bin/env python3
"""
Module for pooling back propagation.
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network.

    Args:
        dA: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
            partial derivatives with respect to the output of the pooling layer
            m is the number of examples
            h_new is the height of the output
            w_new is the width of the output
            c is the number of channels
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c) containing the
                output of the previous layer
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
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
        The partial derivatives with respect to the previous layer (dA_prev)
    """
    m, h_new, w_new, c = dA.shape
    _, h_prev, w_prev, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Initialize gradient
    dA_prev = np.zeros_like(A_prev)

    # Compute gradients
    for i in range(h_new):
        for j in range(w_new):
            # Calculate starting positions
            h_start = i * sh
            w_start = j * sw

            for k in range(c):
                # Extract the region of interest
                region = A_prev[:, h_start:h_start+kh, w_start:w_start+kw, k]

                if mode == 'max':
                    # Create mask for max pooling
                    # Find the maximum value in the region
                    max_val = np.max(region, axis=(1, 2), keepdims=True)
                    mask = (region == max_val).astype(float)
                    # Distribute gradient to the max position
                    dA_prev[:, h_start:h_start+kh, w_start:w_start+kw, k] += (
                        mask * dA[:, i, j, k].reshape(-1, 1, 1)
                    )
                elif mode == 'avg':
                    # Distribute gradient equally for average pooling
                    avg_gradient = dA[:, i, j, k] / (kh * kw)
                    dA_prev[:, h_start:h_start+kh, w_start:w_start+kw, k] += (
                        avg_gradient.reshape(-1, 1, 1)
                    )

    return dA_prev
