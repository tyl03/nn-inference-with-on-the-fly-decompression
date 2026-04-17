"""
Zstd level sweep experiment.

What it does:
- Load trained FCN checkpoint
- Apply global pruning at a chosen amount (default 0.85) and make it permanent
- Export weights/bias layer-by-layer with Zstd using different compression levels
- Measure:
    - compressed payload bytes
    - compression time
    - decompression time (decompress all Linear layers once)
"""

from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt

from nn_compression.exp_utils import build_model, get_device, load_weights
from nn_compression.export_compressed import (
    decompress_linear_layer,
    estimate_compressed_payload_bytes,
    export_fcn_to_compressed,
)
from nn_compression.pruning import (
    global_magnitude_prune_linear_layers,
    make_pruning_permanent,
)


def _measure_decompression_time(compressed: dict) -> float:
    """
    Decompress all Linear layers once and return elapsed time (seconds).
    """
    t0 = time.perf_counter()
    for entry in compressed["layers"]:
        if entry["type"] == "linear":
            _W, _b = decompress_linear_layer(entry)

    t1 = time.perf_counter()
    return t1 - t0


def main():
    device = get_device()

    ckpt_path = "fcn_mnist_best.pt"
    prune_amount = 0.85

    # Level amounts
    levels = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
    ]
    # levels = [5, 6, 7, 8, 9]

    # Load model once
    model = build_model(device)
    load_weights(model, ckpt_path, device)

    # Apply pruning
    global_magnitude_prune_linear_layers(model, amount=prune_amount)
    make_pruning_permanent(model)

    model = model.to("cpu")  # compression/decompression will be on CPU

    results = []

    for lvl in levels:
        # Compression
        t0 = time.perf_counter()
        packed = export_fcn_to_compressed(model, zstd_level=lvl)
        t1 = time.perf_counter()
        compress_s = t1 - t0

        # Size estimation
        payload_bytes = estimate_compressed_payload_bytes(packed)

        # Decompression time estimation
        decompress_s = _measure_decompression_time(packed)

        results.append((lvl, payload_bytes, compress_s, decompress_s))

    # Print results
    print("\nZstd Level Sweep (prune_amount = {:.2f})\n".format(prune_amount))
    header = f"{'level':>6} | {'payload_kb':>11} | {'compress_ms':>12} | {'decompress_ms':>13}"
    print(header)
    print("-" * len(header))

    for lvl, payload_bytes, compress_s, decompress_s in results:
        payload_kb = payload_bytes / 1024.0
        compress_ms = compress_s * 1000.0
        decompress_ms = decompress_s * 1000.0
        print(
            f"{lvl:6d} | {payload_kb:11.2f} | {compress_ms:12.2f} | {decompress_ms:13.2f}"
        )

    # Plots
    levels_list = [r[0] for r in results]
    payload_kb_list = [r[1] / 1024.0 for r in results]
    compress_ms_list = [r[2] * 1000.0 for r in results]
    decompress_ms_list = [r[3] * 1000.0 for r in results]

    os.makedirs("results/zstd", exist_ok=True)

    # Plot 1: Payload size vs Zstd level
    plt.figure()
    plt.plot(levels_list, payload_kb_list, marker="o")
    plt.xlabel("zstd level")
    plt.ylabel("compressed payload (KB)")
    plt.title("Zstd level vs compressed payload size")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/zstd/zstd_payload_vs_level.pdf", bbox_inches="tight")

    # Plot 2: Compression/Decompression time vs Zstd level
    plt.figure()
    plt.plot(levels_list, compress_ms_list, marker="o", label="compress (ms)")
    plt.plot(levels_list, decompress_ms_list, marker="o", label="decompress (ms)")
    plt.xlabel("zstd level")
    plt.ylabel("time (ms)")
    plt.title("Zstd level vs compression / decompression time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/zstd/zstd_time_vs_level.pdf", bbox_inches="tight")

    print("Saved plots: zstd_payload_vs_level.pdf and zstd_time_vs_level.pdf")


if __name__ == "__main__":
    main()
