"""
GPU/CPU device detection and memory monitoring utilities.

Provides utilities for device selection and GPU memory tracking for PyTorch training.
"""

import logging
from typing import Optional

import torch


logger = logging.getLogger(__name__)


def get_device(device_id: Optional[int] = None, prefer_cuda: bool = True) -> torch.device:
    """
    Get the appropriate device for PyTorch operations.

    Detects and logs available GPU/CPU devices. Logs device properties for debugging.

    Args:
        device_id: Specific GPU device ID to use (0-indexed). If None, uses GPU 0 if available.
                   If not None but GPU unavailable, falls back to CPU.
        prefer_cuda: If True (default), prefers CUDA if available. If False, uses CPU.

    Returns:
        torch.device object set to the selected device.

    Examples:
        >>> device = get_device()  # Auto-detect: GPU if available, else CPU
        >>> device = get_device(device_id=1)  # Use GPU 1
        >>> device = get_device(prefer_cuda=False)  # Force CPU
    """
    if prefer_cuda and torch.cuda.is_available():
        if device_id is None:
            device_id = 0

        if device_id >= torch.cuda.device_count():
            logger.warning(
                f"Requested GPU {device_id} but only {torch.cuda.device_count()} "
                f"available. Using GPU 0."
            )
            device_id = 0

        device = torch.device(f"cuda:{device_id}")
        gpu_name = torch.cuda.get_device_name(device_id)
        gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / 1e9

        logger.info(f"Using GPU {device_id}: {gpu_name}")
        logger.info(f"Total GPU Memory: {gpu_memory:.1f} GB")

        # Log number of available GPUs
        if torch.cuda.device_count() > 1:
            logger.info(f"Total GPUs available: {torch.cuda.device_count()}")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS (Metal Performance Shaders)")

    else:
        device = torch.device("cpu")
        logger.info("Using CPU (CUDA not available or disabled)")

    return device


def get_device_name() -> str:
    """
    Get human-readable name of the current device.

    Returns:
        String describing the device (e.g., "NVIDIA GPU: Tesla V100" or "CPU").
    """
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        return f"NVIDIA GPU {device_id}: {torch.cuda.get_device_name(device_id)}"
    else:
        return "CPU"


def log_gpu_memory(prefix: str = "") -> None:
    """
    Log current GPU memory usage.

    Logs allocated and reserved memory. Only logs if CUDA is available.

    Args:
        prefix: Optional prefix string for the log message.

    Examples:
        >>> log_gpu_memory()
        >>> log_gpu_memory(prefix="Epoch 10")
    """
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9

    msg = f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    if max_allocated > 0:
        msg += f", {max_allocated:.2f}GB peak"

    if prefix:
        msg = f"{prefix} | {msg}"

    logger.info(msg)


def reset_gpu_memory() -> None:
    """
    Clear GPU cache and reset memory stats.

    Useful between epochs or after OOM situations. Only takes effect if CUDA is available.

    Examples:
        >>> reset_gpu_memory()  # Between epochs
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.debug("GPU memory cache cleared")


def get_gpu_memory_stats() -> dict:
    """
    Get detailed GPU memory statistics.

    Returns:
        Dictionary with memory stats. Returns empty dict if CUDA unavailable.

    Examples:
        >>> stats = get_gpu_memory_stats()
        >>> print(f"Using {stats['allocated']:.2f}GB")
    """
    if not torch.cuda.is_available():
        return {}

    return {
        'allocated': torch.cuda.memory_allocated() / 1e9,
        'reserved': torch.cuda.memory_reserved() / 1e9,
        'max_allocated': torch.cuda.max_memory_allocated() / 1e9,
        'total': torch.cuda.get_device_properties(0).total_memory / 1e9,
    }
