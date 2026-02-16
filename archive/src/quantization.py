"""
Uniform symmetric quantization for weights (range [-127,127]).

We store:
- int8 weights W_q
- float scale s_w  (per layer)

During inference:
- dequantize: W_fp = W_q * s_w
- compute the layer in FP32
"""

from __future__ import annotations
import torch

INT8_MIN = -127
INT8_MAX = 127


# Force values to stay inside the valid integar range of the quantized representation
def clamp(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, INT8_MIN, INT8_MAX)


def compute_scale(x: torch.Tensor) -> float:
    """
    - alpha = max(abs(x))
    - scale = alpha / 127
    """
    alpha = x.detach().abs().max().item()
    return (alpha / 127.0) if alpha != 0.0 else 1.0


def symmetric_quantization(x: torch.Tensor, scale: float) -> torch.Tensor:
    """
    x_q = clip(round(x/scale), -127, 127) as int8
    """
    x_q = torch.round(x.detach() / scale).to(torch.int32)
    x_q = clamp(x_q).to(torch.int8)
    return x_q


def symmetric_dequantization(x_q: torch.Tensor, scale: float) -> torch.Tensor:
    """
    x_fp ≈ x_q * scale  (returns FP32 tensor)
    """
    return x_q.to(torch.float32) * float(scale)