#!/usr/bin/env python3
"""Module that defines a single neuron for binary classification"""
import numpy as np


class Neuron:
    """A single neuron performing binary classification"""

    def __init__(self, nx):
        """
        Initialize the neuron with private attributes

        Args:
            nx: number of input features to the neuron

        Raises:
            TypeError: if nx is not an integer
            ValueError: if nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data

        Returns:
            The activated output (self.__A)
        """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y: numpy.ndarray with shape (1, m) with correct labels
            A: numpy.ndarray with shape (1, m) with activated output

        Returns:
            The cost
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
            Y: numpy.ndarray with shape (1, m) with correct labels

        Returns:
            Tuple of (predictions, cost)
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
            Y: numpy.ndarray with shape (1, m) with correct labels
            A: numpy.ndarray with shape (1, m) with activated output
            alpha: learning rate
        """
        m = Y.shape[1]
        dz = A - Y
        dW = np.matmul(dz, X.T) / m
        db = np.sum(dz) / m
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
            Y: numpy.ndarray with shape (1, m) with correct labels
            iterations: number of iterations to train over
            alpha: learning rate

        Raises:
            TypeError: if iterations is not an integer or alpha not float
            ValueError: if iterations or alpha are not positive

        Returns:
            Evaluation of training data after iterations
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
