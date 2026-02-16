"""
Unit tests for Huffman coding on int8 tensors.

What these tests check:
- compressing then decompressing an int8 tensor returns identical values
- the compressed package has expected fields and types
"""

import torch

from src.huffman import (
    huff_compress_int8_tensor,
    huff_decompress_int8_tensor,
)


def test_huffman_int8_compress_and_decompress_is_lossless():
    torch.manual_seed(0)

    # simulate quantized weights: int8 values in [-127,127]
    W_q = torch.randint(low=-127, high=128, size=(128, 64), dtype=torch.int8)

    compressed = huff_compress_int8_tensor(W_q)
    W_q_decompressed = huff_decompress_int8_tensor(compressed)

    # must be exact equality for int8 tensors
    assert torch.equal(W_q.cpu(), W_q_decompressed)


def test_compressed_package_has_expected_fields_and_types():
    torch.manual_seed(0)

    W_q = torch.randint(low=-127, high=128, size=(10, 10), dtype=torch.int8)
    compressed = huff_compress_int8_tensor(W_q)

    assert "shape" in compressed
    assert "encoded" in compressed
    assert "freqs" in compressed
    assert "eof" in compressed
    assert "dtype" in compressed

    assert isinstance(compressed["shape"], tuple)
    assert isinstance(compressed["encoded"], (bytes, bytearray))
    assert isinstance(compressed["freqs"], dict)
    assert compressed["dtype"] == "int8"