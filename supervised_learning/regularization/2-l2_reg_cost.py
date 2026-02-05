#!/usr/bin/env python3
"""Module for calculating L2 regularization cost in TensorFlow."""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: tensor containing the cost of the network without L2
              regularization
        model: Keras model that includes layers with L2 regularization

    Returns:
        Tensor containing the total cost for each layer of the network,
        accounting for L2 regularization
    """
    costs = []
    accumulated_cost = cost

    for loss in model.losses:
        accumulated_cost = accumulated_cost + loss
        costs.append(accumulated_cost)

    return tf.stack(costs)
