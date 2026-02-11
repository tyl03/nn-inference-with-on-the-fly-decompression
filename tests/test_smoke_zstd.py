from src.zstd_utils import zstd_compress, zstd_decompress


def test_zstd_compress_decompress():
    original = b"hello world" * 1000  # make it larger to see compression benefits
    compressed = zstd_compress(original, level=3)
    decompressed = zstd_decompress(compressed)
    assert original == decompressed