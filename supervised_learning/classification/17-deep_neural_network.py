#!/usr/bin/env python3
"""Defines DeepNeuralNetwork performing binary classification"""


import numpy as np


class DeepNeuralNetwork:
    """Deep neural network class"""

    def __init__(self, nx, layers):
        """Constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for layer in range(1, self.__L + 1):
            if layer == 1:
                self.__weights["W%d" % layer] = (
                    np.random.randn(layers[layer - 1], nx) *
                    np.sqrt(2. / nx)
                )
            else:
                self.__weights["W%d" % layer] = (
                    np.random.randn(layers[layer - 1], layers[layer - 2]) *
                    np.sqrt(2. / layers[layer - 2])
                )
            self.__weights["b%d" % layer] = np.zeros((layers[layer - 1], 1))

    @property
    def L(self):
        """L getter"""
        return self.__L

    @property
    def cache(self):
        """cache getter"""
        return self.__cache

    @property
    def weights(self):
        """weights getter"""
        return self.__weights
