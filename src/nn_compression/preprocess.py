"""
Input preprocessing utilities.

Purpose:
- Ensure that input data is consistently prepared before being fed into the model.
- Guarantee that training, evaluation, and blockwise inference all use the
  exact same preprocessing logic.

Why this exists:
- In many pipelines, preprocessing (such as scaling or normalization) is hidden
  inside dataset transforms.
- If inference receives raw tensors without the same preprocessing, model
  outputs can be incorrect.

What this module does:
- Converts inputs to float32 (matching the stored FP32 weights).
- Provides a single place where input scaling or normalization can be added
  if needed.

Note:
- If the dataset already uses torchvision.transforms.ToTensor(), images are
  automatically scaled from [0,255] to [0,1], so additional scaling should NOT
  be applied here unless explicitly required.
"""

import torch


def preprocess_input(x: torch.Tensor) -> torch.Tensor:
    """
    Central place for input scaling / normalization.
    """
    # Always use FP32
    x = x.to(dtype=torch.float32)
    return x
