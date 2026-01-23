#!/usr/bin/env python3

import numpy as np

Deep28 = __import__('28-deep_neural_network').DeepNeuralNetwork
one_hot_encode = __import__('24-one_hot_encode').one_hot_encode
one_hot_decode = __import__('25-one_hot_decode').one_hot_decode

lib = np.load('../data/MNIST.npz')
X_train_3D = lib['X_train']
Y_train = lib['Y_train']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
Y_train_one_hot = one_hot_encode(Y_train, 10)

np.random.seed(0)
deep_sig = Deep28(X_train.shape[0], [128, 64, 10], activation='sig')
deep_tanh = Deep28(X_train.shape[0], [128, 64, 10], activation='tanh')

print("Sigmoid activation test:")
print(deep_sig.activation)

print("\nTanh activation test:")
print(deep_tanh.activation)

print("\nTraining with tanh for 100 iterations:")
A, cost = deep_tanh.train(X_train, Y_train_one_hot, iterations=100, 
                          alpha=0.05, verbose=True, graph=False, step=25)
A_decoded = one_hot_decode(A)
accuracy = np.sum(Y_train == A_decoded) / Y_train.shape[0] * 100
print("Train accuracy: {}%".format(accuracy))
