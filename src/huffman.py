import torch
import numpy as np
from collections import Counter
from dahuffman import HuffmanCodec


def tensor_to_symbols(t: torch.Tensor):
    a = t.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
    shape = a.shape
    symbols = a.view(np.uint32).reshape(-1).tolist()
    return symbols, shape


def symbols_to_tensor(symbols, shape):
    a = np.array(symbols, dtype=np.uint32).view(np.float32).reshape(shape)
    return torch.from_numpy(a)


def huffman_compress_tensor(t: torch.Tensor):
    symbols, shape = tensor_to_symbols(t)
    
    # For each distinct symbol, it counts how many times it appears in the data.
    freqs = Counter(symbols)
    
    # From the given symbol frequencies, we build the Huffman code table.
    codec = HuffmanCodec.from_frequencies(freqs)
    
    # Replace each symbol in the original data with its corresponding Huffman code and concatenate those bits, resulting in a compressed bitstream.
    # That bitstream is then packed into bytes.
    encoded = codec.encode(symbols)
    
    compressed_tensor = {
        "shape": shape, # original tensor shape
        "encoded": encoded, # compressed bitstream (bytes)
        "freqs": freqs, # codebook source
    }
    
    return compressed_tensor
    
    
def huffman_decompress_tensor(compressed_tensor):
    codec = HuffmanCodec.from_frequencies(compressed_tensor["freqs"])
    symbols = codec.decode(compressed_tensor["encoded"])
    return symbols_to_tensor(symbols, compressed_tensor["shape"])