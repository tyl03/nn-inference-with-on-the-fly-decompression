import torch

from src.nn_compression.tensor_bytes_utils import from_fp32_bytes, to_fp32_bytes


def test_fp32_bytes_roundtrip():
    # Create a random tensor
    original = torch.randn(10, 7, dtype=torch.float32)

    # Convert to FP32 bytes
    data = to_fp32_bytes(original)

    # Convert back from bytes
    reconstructed = from_fp32_bytes(data, tuple(original.shape))

    # Check that the original and reconstructed tensors are exactly the same
    assert reconstructed.dtype == torch.float32
    assert reconstructed.shape == original.shape
    assert torch.equal(original.cpu(), reconstructed.cpu())
