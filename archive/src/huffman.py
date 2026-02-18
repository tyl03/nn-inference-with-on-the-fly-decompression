"""
Huffman utilities for int8 tensors.

We Huffman-compress int8 weights by treating each byte as a symbol in [0,255].
This works well after quantization because the alphabet is small and repetitive.
"""

from __future__ import annotations

import pickle
from collections import Counter

import numpy as np
import torch
from dahuffman import HuffmanCodec

HUFF_EOF = -1  # Special symbol to indicate end-of-data in Huffman coding


def huff_compress_int8_tensor(t: torch.Tensor) -> dict:
    """
    Huffman-compress an int8 tensor by viewing it as uint8 symbols (0..255).
    Returns a plain dict suitable for torch.save().
    """
    if t.dtype != torch.int8:
        raise TypeError(f"Expected int8 tensor, got {t.dtype}")

    a_u8 = t.detach().cpu().contiguous().view(torch.uint8).numpy()
    shape = tuple(t.shape)
    symbols = a_u8.reshape(-1).tolist()  # list[int] 0..255

    freqs = Counter(symbols)
    codec = HuffmanCodec.from_frequencies(freqs, eof=HUFF_EOF)
    encoded = codec.encode(symbols)

    return {
        "shape": shape,
        "encoded": encoded,
        "freqs": dict(freqs),
        "eof": HUFF_EOF,
        "dtype": "int8",
    }


def huff_decompress_int8_tensor(pkg: dict) -> torch.Tensor:
    """
    Decode a Huffman-compressed int8 tensor package back to torch.int8.
    """
    if pkg.get("dtype") != "int8":
        raise ValueError("Package dtype is not int8")

    freqs = Counter({int(k): int(v) for k, v in pkg["freqs"].items()})
    eof = int(pkg.get("eof", HUFF_EOF))
    codec = HuffmanCodec.from_frequencies(freqs, eof=eof)

    symbols = codec.decode(pkg["encoded"])  # list[int] 0..255
    a_u8 = np.array(symbols, dtype=np.uint8).reshape(pkg["shape"])
    a_i8 = a_u8.view(np.int8)
    return torch.from_numpy(a_i8)


def estimate_huff_pkg_bytes(pkg: dict) -> int:
    """
    Logical estimate: encoded bytes + serialized freqs + serialized shape.
    """
    encoded_bytes = len(pkg["encoded"])
    freqs_bytes = len(pickle.dumps(pkg["freqs"], protocol=pickle.HIGHEST_PROTOCOL))
    shape_bytes = len(pickle.dumps(pkg["shape"], protocol=pickle.HIGHEST_PROTOCOL))
    return encoded_bytes + freqs_bytes + shape_bytes
