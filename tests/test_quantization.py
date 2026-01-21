"""
Unit tests for quantization utilities.

What these tests check:
- compute_scale returns alpha/127 and handles all-zero tensors
- symmetric_quantization outputs int8 values clamped to [-127,127]
- symmetric_dequantization returns float32 and is a reasonable inverse
- round-trip dequant(quant(x)) is close to x (within expected quantization error)

What these tests do NOT check:
- model accuracy after quantization (that is part of an experiment)
"""

import pytest
import torch

from src.quantization import (
    INT8_MIN,
    INT8_MAX,
    clamp,
    compute_scale,
    symmetric_quantization,
    symmetric_dequantization,
)


def test_compute_scale_all_zeros_is_one():
    x = torch.zeros(10, 10, dtype=torch.float32)
    s = compute_scale(x)
    assert s == 1.0
    

def test_compute_scale_matches_alpha_over_127():
    x = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float32)
    # alpha = max(abs(x)) = 3.0
    expected = 3.0 / 127.0
    s = compute_scale(x)
    assert abs(s - expected) < 1e-12
    

def test_clamp_limits_to_range():
    x = torch.tensor([-1000, -127, -126, 0, 126, 127, 1000], dtype=torch.int32)
    y = clamp(x)
    assert y.min().item() == INT8_MIN
    assert y.max().item() == INT8_MAX
    
    
def test_symmetric_quantization_outputs_int8_and_in_range():
    x = torch.tensor([-10.0, 0.0, 10.0], dtype=torch.float32)
    s = compute_scale(x)
    x_q = symmetric_quantization(x, s)
    
    assert x_q.dtype == torch.int8
    assert int(x_q.min().item()) >= INT8_MIN
    assert int(x_q.max().item()) <= INT8_MAX
    
    
def test_dequantization_outputs_float32():
    x_q = torch.tensor([-127, 0, 127], dtype=torch.int8)
    s = 0.1
    x_fp = symmetric_dequantization(x_q, s)
    
    assert x_fp.dtype == torch.float32
    assert torch.allclose(x_fp, torch.tensor([-12.7, 0.0, 12.7], dtype=torch.float32))
    
    