"""
Tests for verifying correctness of blockwise inference.

What this test checks:
- Blockwise inference produces the same output as the original PyTorch model.

How it works:
- A model is exported into compressed blockwise format.
- The same input batch is passed through:
    1) The original model
    2) The blockwise inference pipeline
- The outputs are compared.

Expected result:
- Outputs should be numerically identical or extremely close.
"""

import torch

from src.nn_compression.blockwise_export_compressed import export_fcn_to_compressed
from src.nn_compression.blockwise_inference import blockwise_forward
from src.nn_compression.fcn import FCN


@torch.no_grad()
def test_blockwise_forward_matches_pytorch():
    torch.manual_seed(0)

    device = torch.device("cpu")

    model = FCN(in_dim=28 * 28, hidden_dims=[64], out_dim=10).to(device).eval()

    compressed = export_fcn_to_compressed(
        model,
        block_size=16,
        zstd_level=3,
    )

    # Fake MNIST like batch
    x = torch.randn(8, 1, 28, 28)

    y_ref = model(x)
    y_blockwise = blockwise_forward(compressed, x, device)

    max_abs_diff = (y_ref - y_blockwise).abs().max().item()

    assert max_abs_diff < 1e-5
