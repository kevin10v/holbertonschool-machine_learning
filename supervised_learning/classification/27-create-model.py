#!/usr/bin/env python3

import numpy as np

Deep = __import__('27-deep_neural_network').DeepNeuralNetwork
one_hot_encode = __import__('24-one_hot_encode').one_hot_encode

lib = np.load('../data/MNIST.npz')
X_train_3D = lib['X_train']
Y_train = lib['Y_train']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
Y_train_one_hot = one_hot_encode(Y_train, 10)

np.random.seed(0)
deep = Deep(X_train.shape[0], [128, 64, 10])
A, cost = deep.train(X_train, Y_train_one_hot, iterations=100, 
                     alpha=0.05, verbose=True, graph=False, step=10)
print("Saving model...")
deep.save('27-saved')
print("Model saved to 27-saved.pkl")
