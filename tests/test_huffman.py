"""
Unit tests for the Huffman coding implementation.

What these tests check:
- the conversion of tensors to symbols and back is consistent
- compressing and then decompressing a tensor results in the original tensor
"""

import torch

from src.huffman import (
    tensor_to_symbols,
    symbols_to_tensor,
    huffman_compress_tensor,
    huffman_decompress_tensor,
)


def test_tensor_to_symbols_and_back_is_lossless():
    torch.manual_seed(0)
    
    W = torch.randn(64, 32, dtype=torch.float32)
    
    symbols, shape = tensor_to_symbols(W)
    W_reconstructed = symbols_to_tensor(symbols, shape)
    
    assert torch.allclose(W.cpu(), W_reconstructed)
    
    
def test_huffman_compress_and_decompress_is_lossless():
    torch.manual_seed(0)
    
    W = torch.randn(128, 64, dtype=torch.float32)
    
    compressed = huffman_compress_tensor(W)
    W_decompressed = huffman_decompress_tensor(compressed)
    
    assert torch.equal(W.cpu(), W_decompressed)
    
    
def test_compressed_tesnor_has_expected_fields_and_types():
    torch.manual_seed(0)
    
    W = torch.randn(10, 10, dtype=torch.float32)
    
    compressed = huffman_compress_tensor(W)
    
    assert "shape" in compressed
    assert "encoded" in compressed
    assert "freqs" in compressed
    
    assert isinstance(compressed["shape"], tuple)
    assert isinstance(compressed["encoded"], (bytes, bytearray))
    # Counter is a subclass of dict, so this also checks that freqs is a dict-like object
    assert isinstance(compressed["freqs"], dict)