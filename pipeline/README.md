# Pipeline Projects

## Description
This directory contains projects related to data pipelines and data processing in machine learning. These projects focus on data preprocessing, augmentation, and pipeline construction - essential skills for preparing data for machine learning models.

## Projects

### 1. Data Augmentation
**Directory:** `data_augmentation/`

Implementation of various image data augmentation techniques using TensorFlow. Data augmentation is crucial for:
- Increasing training dataset size artificially
- Improving model generalization
- Reducing overfitting
- Making models robust to variations

**Techniques Implemented:**
- Horizontal flipping
- Random cropping
- Image rotation (90° counter-clockwise)
- Random contrast adjustment
- Random brightness adjustment
- Hue adjustment

**Key Technologies:**
- TensorFlow 2.15
- TensorFlow Datasets
- tf.image API

## Requirements
- Python 3.9
- TensorFlow 2.15
- NumPy 1.25.2
- Ubuntu 20.04 LTS

## Learning Objectives

### Data Augmentation
- Understanding data augmentation concepts and benefits
- Implementing geometric transformations (flip, crop, rotate)
- Implementing color space transformations (brightness, contrast, hue)
- Using TensorFlow's image processing functions
- Knowing when and how to apply augmentation techniques

### Pipeline Design Principles
- **Modularity**: Each transformation is a separate, reusable function
- **Efficiency**: Using TensorFlow operations for GPU acceleration
- **Flexibility**: Functions accept parameters for customization
- **Reproducibility**: Setting seeds for consistent results

## Best Practices

### When to Use Data Augmentation
1. **Limited Training Data**: When you don't have enough labeled data
2. **Overfitting**: When model memorizes training data
3. **Real-world Variability**: To handle diverse input conditions
4. **Class Imbalance**: To balance underrepresented classes

### Common Augmentation Strategies
- **Geometric**: Rotation, flipping, cropping, scaling
- **Color Space**: Brightness, contrast, saturation, hue
- **Noise Addition**: Gaussian noise, salt-and-pepper noise
- **Advanced**: Mixup, Cutout, AutoAugment

### Augmentation Guidelines
- **Don't break semantic meaning**: Augmentations should preserve labels
- **Match real-world conditions**: Apply transformations that reflect actual use cases
- **Balance augmentation**: Too much can hurt performance
- **Test impact**: Measure performance with and without augmentation

## Usage Examples

### Basic Image Augmentation Pipeline
```python
import tensorflow as tf
from data_augmentation import flip_image, crop_image, change_brightness

# Load image
image = tf.io.read_file('image.jpg')
image = tf.image.decode_jpeg(image)

# Apply augmentations
image = flip_image(image)
image = crop_image(image, (200, 200, 3))
image = change_brightness(image, 0.3)
```

### Creating an Augmentation Pipeline
```python
def augment_pipeline(image):
    """Apply multiple augmentations to an image."""
    image = flip_image(image)
    image = change_contrast(image, 0.5, 2.0)
    image = change_brightness(image, 0.2)
    return image

# Apply to dataset
dataset = dataset.map(lambda x, y: (augment_pipeline(x), y))
```

## Project Structure
```
pipeline/
├── README.md                      # This file
└── data_augmentation/            # Data augmentation techniques
    ├── 0-flip.py                 # Horizontal flip
    ├── 1-crop.py                 # Random crop
    ├── 2-rotate.py               # 90° rotation
    ├── 3-contrast.py             # Contrast adjustment
    ├── 4-brightness.py           # Brightness adjustment
    ├── 5-hue.py                  # Hue adjustment
    └── README.md                 # Detailed documentation
```

## Resources
- [TensorFlow Image Processing Guide](https://www.tensorflow.org/api_docs/python/tf/image)
- [Data Augmentation Best Practices](https://www.tensorflow.org/tutorials/images/data_augmentation)
- [AutoAugment Paper](https://arxiv.org/abs/1805.09501)
- [A Complete Guide to Data Augmentation](https://towardsdatascience.com/complete-guide-to-data-augmentation-for-computer-vision-1abe4063ad07)

## Future Projects
Potential future additions to this pipeline directory:
- ETL pipelines
- Data validation and cleaning
- Feature engineering pipelines
- Real-time data streaming
- Data versioning and management
- Pipeline orchestration with Apache Airflow/Kubeflow

## Author
Holberton School Machine Learning Specialization
