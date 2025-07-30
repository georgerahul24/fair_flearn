# PyTorch Migration Guide

This repository has been converted from TensorFlow to PyTorch for better compatibility and ease of use.

## Changes Made

### Model Files Converted:
1. `flearn/models/adult/lr.py` - Logistic Regression (99 features)
2. `flearn/models/fmnist/lr.py` - Logistic Regression (784 features for 28x28 images)
3. `flearn/models/synthetic/mclr.py` - Logistic Regression (60 features)
4. `flearn/models/vehicle/svm.py` - SVM with hinge loss (100 features)

### Key Changes:

#### Dependencies:
- **Old**: `tensorflow-gpu==1.10`
- **New**: `torch>=1.9.0`, `torchvision>=0.10.0`

#### Model Structure:
- Each model now contains a PyTorch `nn.Module` class for the neural network
- The main `Model` class provides the same API interface as before
- All methods (`set_params`, `get_params`, `get_gradients`, `solve_inner`, etc.) maintain the same signatures

#### New Features:
- Automatic GPU detection and usage (`cuda` if available)
- Better random seed management
- Cleaner code structure with separate model definition

#### Utility Changes:
- Added `flearn/utils/torch_utils.py` with PyTorch equivalents of TensorFlow utilities
- Model size calculation, gradient processing, and other utilities converted

## Usage

The API remains exactly the same. Your existing training scripts should work without modification:

```python
from flearn.models.adult.lr import Model

# Initialize model (same as before)
model = Model(num_classes=2, q=None, optimizer='sgd', seed=1)

# All methods work the same
params = model.get_params()
model.set_params(params)
loss = model.get_loss(data)
num_samples, grads = model.get_gradients(data, model_len)
solution, comp = model.solve_inner(data, num_epochs=5)
```

## Installation

Update your environment with:

```bash
pip install torch>=1.9.0 torchvision>=0.10.0
```

Or install from the updated requirements.txt:

```bash
pip install -r requirements.txt
```

## Benefits of PyTorch Migration

1. **Modern Framework**: PyTorch is actively maintained and widely adopted
2. **Better Debugging**: Dynamic computation graphs make debugging easier
3. **GPU Support**: Seamless GPU acceleration with CUDA
4. **Memory Efficiency**: Better memory management
5. **Ecosystem**: Access to latest research implementations and pretrained models

## Backward Compatibility

- All public APIs remain the same
- Model parameters can be loaded/saved in the same format
- Training loops require no changes
- Results should be numerically equivalent to the TensorFlow version
