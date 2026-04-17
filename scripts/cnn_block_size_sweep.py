"""
Block size sweep experiment for blockwise CNN inference on MNIST.

Measures for each block size:
- accuracy
- latency (ms/sample)
- estimated peak runtime RAM (KB)

Purpose:
- Study the trade-off between memory usage and inference latency
  for different CNN block sizes.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import torch

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
    blockwise_evaluate_accuracy,
    measure_blockwise_inference_time,
)
from nn_compression.exp_utils import (
    build_model,
    estimate_peak_runtime_blockwise_bytes,
    get_device,
    load_test_loader,
    load_weights,
)


def save_time_and_peak_block_chart(results, out_path: str) -> None:
    """
    Plot:
    - Left axis: ms/sample
    - Right axis: peak runtime (KB)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    block_sizes = [r[0] for r in results]
    peak_runtime_kb = [r[2] / 1024 for r in results]
    ms_sample = [r[3] for r in results]

    x_labels = [str(bs) for bs in block_sizes]
    x = list(range(len(block_sizes)))
    width = 0.4

    fig, ax1 = plt.subplots()

    # Latency bars
    ax1.bar(
        [i - width / 2 for i in x],
        ms_sample,
        width=width,
        color="tab:blue",
        alpha=0.85,
        label="Latency (ms/sample)",
    )

    ax1.set_xlabel("Block size")
    ax1.set_ylabel("Latency (ms/sample)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)

    # Peak runtime bars
    ax2 = ax1.twinx()
    ax2.bar(
        [i + width / 2 for i in x],
        peak_runtime_kb,
        width=width,
        color="tab:orange",
        alpha=0.85,
        label="Peak runtime (KB)",
    )

    ax2.set_ylabel("Peak runtime (KB)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    ax1.set_title("CNN Block Size Sweep: ms/sample vs Peak Runtime RAM")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    device = get_device()
    infer_device = torch.device("cpu")
    test_loader = load_test_loader(batch_size=1)
    batch_size = 1

    if batch_size is None or batch_size <= 0:
        raise ValueError("Invalid test loader batch size.")

    ckpt_path = "cnn_mnist_best.pt"
    zstd_level = 16

    # These are output-channel block sizes for convolution/linear layers
    block_sizes = [4, 8, 16, 32, 64]
    # block_sizes = [2, 4, 6, 8, 10, 12, 14, 16]

    # Build and load trained CNN
    model = build_model(device, model_type="cnn")
    load_weights(model, ckpt_path, device)

    os.makedirs("results/zstd/sweeps", exist_ok=True)

    print("\nCNN block size sweep")
    print("-" * 80)
    print(
        f"{'bs':>4} | {'acc':>8} | {'ms/sample':>12} | {'peak_runtime (KB)':>18}"
    )
    print("-" * 80)

    results = []

    for bs in block_sizes:
        save_path = (
            f"results/zstd/sweeps/cnn_mnist_bs{bs}_zstd{zstd_level}.pt"
        )

        packed = export_blockwise(model, zstd_level=zstd_level, block_size=bs)
        save_blockwise(packed, save_path)
        compressed = load_blockwise(save_path)

        # Accuracy
        acc = blockwise_evaluate_accuracy(compressed, test_loader, infer_device)

        # Peak runtime estimate
        peak_runtime_bytes = estimate_peak_runtime_blockwise_bytes(
            model, bs, batch_size
        )

        # Timing
        t_batch = measure_blockwise_inference_time(
            compressed, test_loader, infer_device
        )
        ms_sample = (t_batch * 1000.0) / batch_size

        results.append((bs, acc, peak_runtime_bytes, ms_sample))

        print(
            f"{bs:4d} | {acc:8.4f} | {ms_sample:12.4f} | {peak_runtime_bytes/1024:18.1f}"
        )

    print("-" * 80)

    plot_path = "results/zstd/sweeps/cnn_ms_sample_vs_peakruntime.pdf"
    save_time_and_peak_block_chart(results, plot_path)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()