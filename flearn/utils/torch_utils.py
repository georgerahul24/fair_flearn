import numpy as np
import torch


def model_size(model):
    """Returns the size of the given PyTorch model in bytes

    The size of the model is calculated by summing up the sizes of each
    parameter. The sizes of parameters are calculated by multiplying
    the number of bytes in their dtype with their number of elements.

    Args:
        model: PyTorch model
    Return:
        integer representing size of model (in bytes)
    """
    total_size = 0
    for param in model.parameters():
        total_size += param.numel() * param.element_size()
    return total_size


def process_grad(grads):
    """
    Args:
        grads: list of gradient tensors from PyTorch model
    Return:
        a flattened grad in numpy (1-D array)
    """
    if not grads:
        return np.array([])
    
    # Convert tensors to numpy and flatten
    numpy_grads = []
    for grad in grads:
        if grad is not None:
            numpy_grads.append(grad.cpu().numpy().flatten())
        else:
            # Handle None gradients (shouldn't happen in normal cases)
            continue
    
    if not numpy_grads:
        return np.array([])
    
    # Concatenate all gradients
    client_grads = numpy_grads[0]
    for i in range(1, len(numpy_grads)):
        client_grads = np.append(client_grads, numpy_grads[i])
    
    return client_grads


def cosine_sim(a, b):
    """Returns the cosine similarity between two arrays a and b
    
    Args:
        a: numpy array
        b: numpy array
    
    Return:
        cosine similarity (float)
    """
    if len(a.shape) > 1:
        a = a.flatten()
    if len(b.shape) > 1:
        b = b.flatten()
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def norm_grad(grad_list):
    """Calculate the L2 norm of gradients
    
    Args:
        grad_list: list of gradient arrays
    
    Return:
        L2 norm (float)
    """
    if not grad_list:
        return 0.0
    
    # Flatten and concatenate all gradients
    flat_grad = process_grad(grad_list)
    return np.linalg.norm(flat_grad)


def calculate_flops(model, input_shape):
    """Approximate FLOPs calculation for a model
    
    Args:
        model: PyTorch model
        input_shape: tuple representing input shape (excluding batch dimension)
    
    Return:
        estimated FLOPs (int)
    """
    total_flops = 0
    
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            # For linear layer: FLOPs = input_features * output_features * 2
            total_flops += module.in_features * module.out_features * 2
        elif isinstance(module, torch.nn.Conv2d):
            # For conv layer: FLOPs = output_h * output_w * kernel_h * kernel_w * input_channels * output_channels
            # This is a simplified calculation
            kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            output_elements = module.out_channels
            total_flops += kernel_flops * output_elements
    
    return total_flops
