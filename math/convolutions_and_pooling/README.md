# Convolutions and Pooling

## Description
This project implements fundamental convolution and pooling operations for image processing and convolutional neural networks. All implementations are done from scratch using NumPy, without using `np.convolve`.

## Learning Objectives
- Understanding convolution operations
- Understanding max pooling and average pooling
- Understanding kernels/filters
- Understanding padding (same, valid) and strides
- Working with image channels
- Implementing convolution over images
- Implementing pooling over images

## Requirements
- Python 3.9
- NumPy 1.25.2
- Ubuntu 20.04 LTS
- pycodestyle 2.11.1

## Project Structure

### Task 0: Valid Convolution
**File:** `0-convolve_grayscale_valid.py`

Performs a valid convolution on grayscale images. No padding is applied.

### Task 1: Same Convolution
**File:** `1-convolve_grayscale_same.py`

Performs a same convolution on grayscale images. Padding is applied to maintain the same output dimensions as input.

### Task 2: Convolution with Padding
**File:** `2-convolve_grayscale_padding.py`

Performs convolution on grayscale images with custom padding specified as a tuple `(ph, pw)`.

### Task 3: Strided Convolution
**File:** `3-convolve_grayscale.py`

Performs convolution on grayscale images with configurable stride and padding modes:
- `padding='same'`: Same convolution
- `padding='valid'`: Valid convolution
- `padding=(ph, pw)`: Custom padding

### Task 4: Convolution with Channels
**File:** `4-convolve_channels.py`

Performs convolution on images with multiple channels (e.g., RGB images).

### Task 5: Multiple Kernels
**File:** `5-convolve.py`

Performs convolution on images using multiple kernels, producing multiple output channels.

### Task 6: Pooling
**File:** `6-pool.py`

Performs pooling operations on images:
- Max pooling (`mode='max'`)
- Average pooling (`mode='avg'`)

## Usage

Each file can be tested with the corresponding main file provided in the project description. For example:

```bash
./0-main.py
./1-main.py
# ... and so on
```

## Key Concepts

### Convolution
A mathematical operation that combines two functions to produce a third function. In image processing, it involves sliding a kernel/filter over an image to extract features.

### Padding
Adding zeros around the border of an image to control the output dimensions:
- **Valid**: No padding
- **Same**: Padding to maintain input dimensions

### Stride
The number of pixels the kernel moves at each step during convolution.

### Pooling
A downsampling operation that reduces the spatial dimensions of the feature maps:
- **Max Pooling**: Takes the maximum value in each pooling window
- **Average Pooling**: Takes the average value in each pooling window

## Author
Holberton School Machine Learning Project
