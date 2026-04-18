"""
Reproducible random seed management for PyTorch experiments.

Provides utilities to set seeds across all random sources (Python, NumPy, PyTorch)
to ensure reproducible training runs.
"""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Configures seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - CUDA if available
    - cuDNN backend behavior

    Args:
        seed: Random seed value to set. Should be a non-negative integer.
        deterministic: If True (default), sets PyTorch to deterministic mode
                      which may be slightly slower but guarantees reproducibility.
                      Setting to False allows faster GPU algorithms but may introduce
                      non-determinism.

    Examples:
        >>> set_seed(42)  # Reproducible training
        >>> set_seed(seed=1337, deterministic=False)  # Faster but less deterministic

    Notes:
        - When deterministic=True, torch.cuda.benchmark is set to False
        - cuDNN is configured to use deterministic algorithms
        - Some CUDA operations may not have deterministic implementations
        - Performance may vary slightly depending on GPU operations
    """
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seeds
    torch.manual_seed(seed)

    # Handle CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Set seed for all GPU devices

        if deterministic:
            # Enable deterministic algorithms in cuDNN
            torch.backends.cudnn.deterministic = True
            # Disable benchmarking which can introduce non-determinism
            torch.backends.cudnn.benchmark = False
        else:
            # Allow cuDNN to use fastest algorithms (may be non-deterministic)
            torch.backends.cudnn.benchmark = True


def get_seed_info() -> dict:
    """
    Get information about current random seed configuration.

    Returns:
        Dictionary with seed configuration status for debugging purposes.

    Examples:
        >>> info = get_seed_info()
        >>> print(info['cudnn_deterministic'])
    """
    return {
        'cuda_available': torch.cuda.is_available(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
    }
