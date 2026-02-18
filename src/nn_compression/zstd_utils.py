import zstandard as zstd


def zstd_compress(data: bytes, level: int = 3) -> bytes:
    """Compresses the given data using zstandard."""
    compressor = zstd.ZstdCompressor(level=level)
    return compressor.compress(data)


def zstd_decompress(compressed_data: bytes) -> bytes:
    """Decompresses the given zstandard compressed data."""
    decompressor = zstd.ZstdDecompressor()
    return decompressor.decompress(compressed_data)
