# Data Augmentation

## Description
This project implements various data augmentation techniques using TensorFlow. Data augmentation is a crucial technique in deep learning that helps create more diverse training data by applying transformations to existing images, improving model generalization and reducing overfitting.

## Learning Objectives
- Understanding what data augmentation is
- Knowing when to perform data augmentation
- Understanding the benefits of data augmentation
- Learning various ways to perform data augmentation
- Using TensorFlow's image processing functions for augmentation

## Requirements
- Python 3.9
- TensorFlow 2.15
- NumPy 1.25.2
- TensorFlow Datasets 4.9.2
- Ubuntu 20.04 LTS
- pycodestyle 2.11.1

## Installation
```bash
pip install --user tensorflow-datasets==4.9.2
```

## Project Structure

### Task 0: Flip
**File:** `0-flip.py`

Flips an image horizontally using `tf.image.flip_left_right()`.

**Usage:**
```python
flip_image(image)
```

### Task 1: Crop
**File:** `1-crop.py`

Performs a random crop of an image using `tf.image.random_crop()`.

**Usage:**
```python
crop_image(image, size)
```

### Task 2: Rotate
**File:** `2-rotate.py`

Rotates an image by 90 degrees counter-clockwise using `tf.image.rot90()`.

**Usage:**
```python
rotate_image(image)
```

### Task 3: Contrast
**File:** `3-contrast.py`

Randomly adjusts the contrast of an image using `tf.image.random_contrast()`.

**Usage:**
```python
change_contrast(image, lower, upper)
```

### Task 4: Brightness
**File:** `4-brightness.py`

Randomly changes the brightness of an image using `tf.image.random_brightness()`.

**Usage:**
```python
change_brightness(image, max_delta)
```

### Task 5: Hue
**File:** `5-hue.py`

Changes the hue of an image using `tf.image.adjust_hue()`.

**Usage:**
```python
change_hue(image, delta)
```

## Key Concepts

### What is Data Augmentation?
Data augmentation is a technique to artificially increase the size and diversity of a training dataset by applying various transformations to existing images without collecting new data.

### When to Use Data Augmentation
- When you have limited training data
- To prevent overfitting
- To improve model generalization
- To make models robust to variations in input

### Benefits
- Increases effective training set size
- Reduces overfitting
- Improves model generalization
- Makes models more robust to real-world variations
- Cost-effective alternative to collecting more data

### Common Augmentation Techniques
1. **Geometric Transformations:**
   - Flipping (horizontal/vertical)
   - Rotation
   - Cropping
   - Scaling
   - Translation

2. **Color Space Transformations:**
   - Brightness adjustment
   - Contrast adjustment
   - Hue adjustment
   - Saturation adjustment

3. **Advanced Techniques:**
   - Mixup
   - CutOut
   - AutoAugment
   - RandAugment

## Testing
Each file can be tested with its corresponding main file:
```bash
./0-main.py
./1-main.py
./2-main.py
./3-main.py
./4-main.py
./5-main.py
```

## TensorFlow Image Functions Used
- `tf.image.flip_left_right()` - Horizontal flip
- `tf.image.random_crop()` - Random cropping
- `tf.image.rot90()` - 90-degree rotation
- `tf.image.random_contrast()` - Random contrast adjustment
- `tf.image.random_brightness()` - Random brightness adjustment
- `tf.image.adjust_hue()` - Hue adjustment

## Author
Holberton School Machine Learning Project
