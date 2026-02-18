"""
Utilities for converting tensors to/from raw FP32 bytes.
"""

from __future__ import annotations

import numpy as np
import torch


def to_fp32_bytes(tensor: torch.Tensor) -> bytes:
    """
    Convert a PyTorch tensor to raw FP32 bytes.
    """
    t_cpu = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return t_cpu.numpy().tobytes()


def from_fp32_bytes(data: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    """
    Convert raw FP32 bytes back to a PyTorch tensor with the given shape.
    """
    arr = np.frombuffer(data, dtype=np.float32).reshape(shape)
    return torch.from_numpy(arr.copy())
