"""
Block size sweep experiment (MNIST) for blockwise Zstd FP32 inference.

Measures for each block_size:
- ms/sample (end-to-end latency per input)
- Peak decompressed weight-block bytes (RAM estimate)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from nn_compression.blockwise_export_compressed import (
    export_fcn_to_compressed as export_blockwise,
)
from nn_compression.blockwise_export_compressed import (
    load_compressed as load_blockwise,
)
from nn_compression.blockwise_export_compressed import (
    save_compressed as save_blockwise,
)
from nn_compression.blockwise_inference import (
    measure_blockwise_inference_time,
)
from nn_compression.exp_utils import (
    build_model,
    get_device,
    load_test_loader,
    load_weights,
)
from nn_compression.pruning import (
    global_magnitude_prune_linear_layers,
    make_pruning_permanent,
)


def estimate_peak_decompressed_block_bytes_from_model(model: nn.Module, block_size: int) -> int:
    """
    Peak decompressed weights for blockwise inference (FP32), estimated from model layer shapes.
    For each Linear layer: block_out = min(block_size, out_features), bytes = block_out*in_features*4
    Returns the maximum across layers.
    """
    peak = 0
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            in_features = int(layer.in_features)
            out_features = int(layer.out_features)
            block_out = min(block_size, out_features)
            block_bytes = block_out * in_features * 4  # FP32 bytes
            peak = max(peak, block_bytes)
    return peak


def save_time_and_peak_block_chart(results, out_path: str) -> None:
    """
    Plot:
    - Left axis: ms/sample
    - Right axis: peak_block (KB)
    """

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    block_sizes = [r[0] for r in results]
    peak_kb     = [r[1] / 1024 for r in results]
    ms_sample   = [r[2] for r in results]

    x_labels = [str(bs) for bs in block_sizes]
    x = list(range(len(block_sizes)))
    width = 0.4

    fig, ax1 = plt.subplots()

    # ms/sample bars
    ax1.bar(
        [i - width/2 for i in x],
        ms_sample,
        width=width,
        color="tab:blue",
        alpha=0.85,
        label="Latency (ms/sample)"
    )

    ax1.set_xlabel("Block size")
    ax1.set_ylabel("Latency (ms/sample)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)

    # Peak RAM bars
    ax2 = ax1.twinx()
    ax2.bar(
        [i + width/2 for i in x],
        peak_kb,
        width=width,
        color="tab:orange",
        alpha=0.85,
        label="Peak block (KB)"
    )

    ax2.set_ylabel("Peak block (KB)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    ax1.set_title("Block Size Sweep: ms/sample vs Peak Block RAM")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    device = get_device()
    infer_device = torch.device("cpu")  # blockwise inference on CPU for realistic timing
    test_loader = load_test_loader()
    
    batch_size = test_loader.batch_size
    
    ckpt_path = "fcn_mnist_best.pt"
    prune_amount = 0.85
    zstd_level = 7
    
    block_sizes = [8, 16, 32, 64, 128, 256]
    
    # Build and prune model once
    model = build_model(device)
    load_weights(model, ckpt_path, device)
    
    global_magnitude_prune_linear_layers(model, amount=prune_amount)
    make_pruning_permanent(model)
    
    os.makedirs("results/zstd/sweeps", exist_ok=True)

    print("\nBlock size sweep (essential metrics)")
    print("-" * 60)
    print(f"{'bs':>4} | {'ms/sample':>16} | {'peak_block (KB)':>16}")
    print("-" * 60)

    results = []

    for bs in block_sizes:
        save_path = f"results/zstd/sweeps/fcn_mnist_pruned_{int(prune_amount*100)}_bs{bs}_zstd{zstd_level}.pt"

        packed = export_blockwise(model, zstd_level=zstd_level, block_size=bs)
        save_blockwise(packed, save_path)
        compressed = load_blockwise(save_path)

        peak_block_bytes = estimate_peak_decompressed_block_bytes_from_model(model, bs)

        t_batch = measure_blockwise_inference_time(compressed, test_loader, infer_device)

        ms_sample = (t_batch * 1000.0) / batch_size  # convert s/batch -> ms/sample

        results.append((bs, peak_block_bytes, ms_sample))

        print(f"{bs:4d} | {ms_sample:16.4f} | {peak_block_bytes/1024:16.1f}")

    print("-" * 60)

    plot_path = "results/zstd/sweeps/ms_sample_vs_peakblock.png"
    save_time_and_peak_block_chart(results, plot_path)
    print(f"Saved plot: {plot_path}")
    print("-" * 60)


if __name__ == '__main__':
    main()