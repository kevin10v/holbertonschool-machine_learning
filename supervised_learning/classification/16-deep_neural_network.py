#!/usr/bin/env python3
"""Module that defines a deep neural network"""
import numpy as np


class DeepNeuralNetwork:
    """Deep neural network for binary classification"""

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network

        Args:
            nx: number of input features
            layers: list representing number of nodes in each layer

        Raises:
            TypeError: if nx not int or layers not list of positive ints
            ValueError: if nx < 1 or layers is empty
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for layer in range(1, self.L + 1):
            if layer == 1:
                self.weights[f'W{layer}'] = (
                    np.random.randn(layers[layer - 1], nx) *
                    np.sqrt(2 / nx)
                )
            else:
                self.weights[f'W{layer}'] = (
                    np.random.randn(layers[layer - 1], layers[layer - 2]) *
                    np.sqrt(2 / layers[layer - 2])
                )
            self.weights[f'b{layer}'] = np.zeros((layers[layer - 1], 1))
