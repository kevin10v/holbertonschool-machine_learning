#!/usr/bin/env python3
"""Defines a deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Deep neural network class"""

    def __init__(self, nx, layers):
        """Initialize the deep neural network"""
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
                self.weights["W{}".format(layer)] = np.random.randn(
                    layers[layer - 1], nx) * np.sqrt(2 / nx)
            else:
                self.weights["W{}".format(layer)] = np.random.randn(
                    layers[layer - 1], layers[layer - 2]) * np.sqrt(2 / layers[layer - 2])
            self.weights["b{}".format(layer)] = np.zeros((layers[layer - 1], 1))
