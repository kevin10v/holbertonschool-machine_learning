# Convolutional Neural Networks (CNNs)

## Description
This project implements the fundamental operations of Convolutional Neural Networks (CNNs) from scratch, including forward and backward propagation for convolutional and pooling layers. It also includes an implementation of the classic LeNet-5 architecture using TensorFlow/Keras.

## Learning Objectives
- Understanding convolutional layers and their operations
- Understanding pooling layers (max and average)
- Implementing forward propagation over convolutional and pooling layers
- Implementing back propagation over convolutional and pooling layers
- Building CNNs using TensorFlow and Keras
- Understanding the LeNet-5 architecture

## Requirements
- Python 3.9
- NumPy 1.25.2
- TensorFlow 2.15
- Ubuntu 20.04 LTS
- pycodestyle 2.11.1

## Project Structure

### Task 0: Convolutional Forward Propagation
**File:** `0-conv_forward.py`

Performs forward propagation over a convolutional layer of a neural network.

**Features:**
- Supports 'same' and 'valid' padding
- Configurable stride
- Applies activation function after convolution
- Handles multiple input channels and multiple output filters

**Function:**
```python
def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1))
```

### Task 1: Pooling Forward Propagation
**File:** `1-pool_forward.py`

Performs forward propagation over a pooling layer of a neural network.

**Features:**
- Supports max pooling and average pooling
- Configurable kernel size and stride
- Maintains channel dimension

**Function:**
```python
def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max')
```

### Task 2: Convolutional Back Propagation
**File:** `2-conv_backward.py`

Performs back propagation over a convolutional layer of a neural network.

**Features:**
- Computes gradients with respect to previous layer (dA_prev)
- Computes gradients with respect to kernels (dW)
- Computes gradients with respect to biases (db)
- Handles both 'same' and 'valid' padding
- Supports strided convolutions

**Function:**
```python
def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1))
```

### Task 3: Pooling Back Propagation
**File:** `3-pool_backward.py`

Performs back propagation over a pooling layer of a neural network.

**Features:**
- Max pooling backprop: distributes gradient to max position
- Average pooling backprop: distributes gradient equally
- Handles arbitrary kernel sizes and strides

**Function:**
```python
def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max')
```

### Task 4: LeNet-5 (Keras)
**File:** `5-lenet5.py`

Builds a modified version of the LeNet-5 architecture using Keras.

**Architecture:**
1. Convolutional layer: 6 kernels (5×5), same padding, ReLU
2. Max pooling layer: 2×2 kernel, 2×2 stride
3. Convolutional layer: 16 kernels (5×5), valid padding, ReLU
4. Max pooling layer: 2×2 kernel, 2×2 stride
5. Flatten layer
6. Fully connected layer: 120 nodes, ReLU
7. Fully connected layer: 84 nodes, ReLU
8. Softmax output layer: 10 nodes

**Features:**
- Uses He Normal initialization (seed=0) for reproducibility
- Compiled with Adam optimizer
- Uses categorical crossentropy loss
- Tracks accuracy metric

**Function:**
```python
def lenet5(X)
```

## Usage

### Testing Convolutional Forward Propagation
```bash
./0-main.py
```

### Testing Pooling Forward Propagation
```bash
./1-main.py
```

### Testing Convolutional Back Propagation
```bash
./2-main.py
```

### Testing Pooling Back Propagation
```bash
./3-main.py
```

### Testing LeNet-5
```bash
./5-main.py
```

## Key Concepts

### Convolutional Layer
A convolutional layer applies a set of learnable filters (kernels) to the input. Each filter slides across the input spatially, computing dot products between the filter weights and local regions of the input.

**Key parameters:**
- **Kernel size**: Dimensions of the filter (e.g., 3×3, 5×5)
- **Stride**: Step size for sliding the kernel
- **Padding**: Adding zeros around the input
  - *Same*: Output size matches input size (with stride=1)
  - *Valid*: No padding applied

### Pooling Layer
A pooling layer reduces the spatial dimensions of the input, providing translational invariance and reducing computational complexity.

**Types:**
- **Max Pooling**: Takes the maximum value in each pooling window
- **Average Pooling**: Takes the average value in each pooling window

### Back Propagation in CNNs

**Convolutional Layer:**
- dW: Convolution between input and upstream gradient
- db: Sum of upstream gradient
- dA_prev: Full convolution between upstream gradient and rotated kernel

**Pooling Layer:**
- Max pooling: Gradient flows only to the max position
- Average pooling: Gradient distributed equally

### LeNet-5 Architecture
LeNet-5 is one of the earliest convolutional neural networks, designed by Yann LeCun for handwritten digit recognition. It demonstrated the power of CNNs for image classification tasks.

## References
- [Gradient-Based Learning Applied to Document Recognition (LeNet-5)](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
- [CS231n: Convolutional Neural Networks](http://cs231n.github.io/convolutional-networks/)
- [TensorFlow Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)

## Author
Holberton School Machine Learning Project
