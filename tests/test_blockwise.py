import torch

from src.blockwise_utils import (
    iter_row_blocks,
    num_blocks,
    block_shape,
    compress_weight_blocks_fp32,
    decompress_weight_block_fp32,
    decompress_bias_fp32,
)
from src.zstd_utils import zstd_compress, zstd_decompress
from src.tensor_bytes_utils import to_fp32_bytes, from_fp32_bytes


def test_iter_row_blocks_shapes_and_ranges():
    W = torch.randn(10, 5)
    block_size = 4
    
    blocks = list(iter_row_blocks(W, block_size))
    
    # expected blocks: [0:4], [4:8], [8:10]
    assert len(blocks) == 3
    
    (s0, e0, b0) = blocks[0]
    assert (s0, e0) == (0, 4)
    assert b0.shape == (4, 5)
    
    (s1, e1, b1) = blocks[1]
    assert (s1, e1) == (4, 8)
    assert b1.shape == (4, 5)
    
    (s2, e2, b2) = blocks[2]
    assert (s2, e2) == (8, 10)
    assert b2.shape == (2, 5)
    
    
def test_num_blocks_rounding_up():
    assert num_blocks(10, 4) == 3
    assert num_blocks(8, 4) == 2
    assert num_blocks(1, 4) == 1
    
    
def test_block_shape_matches_actual_slices():
    in_features = 5
    out_features = 10
    block_size = 4
    
    assert block_shape(in_features, out_features, block_size, 0) == (4, 5)
    assert block_shape(in_features, out_features, block_size, 1) == (4, 5)
    assert block_shape(in_features, out_features, block_size, 2) == (2, 5)
    
    
def test_compress_then_decompress_block_exact_match():
    W = torch.randn(10, 5, dtype=torch.float32)
    block_size = 4
    zstd_level = 3
    
    W_blocks_zstd = compress_weight_blocks_fp32(W, block_size=block_size, zstd_level=zstd_level)
    
    entry = {
        "type": "linear",
        "storage": "blockwise",
        "in_features": 5,
        "out_features": 10,
        "block_size": block_size,
        "W_blocks_zstd": W_blocks_zstd,
        "b_zstd": None,
        "bias_shape": None,
    }
    
    # check each block matches the original slice
    blocks = list(iter_row_blocks(W, block_size))
    for block_idx, (start, end, W_block_true) in enumerate(blocks):
        W_block_decompressed = decompress_weight_block_fp32(entry, block_idx)
        assert W_block_decompressed.shape == W_block_true.shape
        assert torch.equal(W_block_true.cpu(), W_block_decompressed.cpu())
        
        
def test_bias_decompress_roundtrup_exact_match():
    b = torch.randn(10, dtype=torch.float32)
    zstd_level = 3
    
    b_raw = to_fp32_bytes(b)
    b_zstd = zstd_compress(b_raw, level=zstd_level)
    
    entry = {
        "type": "linear",
        "storage": "blockwise",
        "in_features": 5,
        "out_features": 10,
        "block_size": 4,
        "W_blocks_zstd": [],
        "b_zstd": b_zstd,
        "bias_shape": tuple(b.shape),
    }
    
    b_decompressed = decompress_bias_fp32(entry)
    assert b_decompressed is not None
    assert b_decompressed.shape == b.shape
    assert torch.equal(b.cpu(), b_decompressed.cpu())