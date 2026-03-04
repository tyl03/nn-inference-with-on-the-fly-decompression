"""
Tests for verifying correctness of blockwise weight compression.

What this test checks:
- Exporting a model with blockwise compression preserves the exact FP32
  weight values after decompression.
- Zstandard compression and tensor byte conversion do not modify data.

How it works:
- A model is exported using blockwise compression.
- Each compressed weight block is decompressed.
- The reconstructed weight matrix is compared with the original weight matrix.

Expected result:
- The reconstructed weights must be bitwise identical to the original weights.
"""

import torch
import torch.nn as nn

from src.nn_compression.blockwise_export_compressed import export_fcn_to_compressed
from src.nn_compression.blockwise_utils import decompress_weight_block_fp32
from src.nn_compression.fcn import FCN


def test_export_decompress_weights_are_identical():
    torch.manual_seed(0)

    model = FCN(in_dim=28 * 28, hidden_dims=[64], out_dim=10).eval()

    compressed = export_fcn_to_compressed(
        model,
        block_size=16,
        zstd_level=3,
    )

    linear_layers = [layer for layer in model.net if isinstance(layer, nn.Linear)]
    compressed_layers = [
        entry for entry in compressed["layers"] if entry["type"] == "linear"
    ]

    assert len(linear_layers) == len(compressed_layers)

    for layer, entry in zip(linear_layers, compressed_layers):
        W = layer.weight.detach().cpu().to(torch.float32).contiguous()

        blocks = []
        for block_idx in range(len(entry["W_blocks_zstd"])):
            blocks.append(decompress_weight_block_fp32(entry, block_idx))

        W_reconstructed = torch.cat(blocks, dim=0).contiguous()

        assert W.shape == W_reconstructed.shape

        # Compare raw bytes (bitwise check)
        assert torch.equal(
            W.view(torch.int8),
            W_reconstructed.view(torch.int8),
        )
