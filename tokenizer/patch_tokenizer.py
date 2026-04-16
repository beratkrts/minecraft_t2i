"""
2x2 Patch Tokenizer for 16x16 4-bit grayscale images.

Pipeline:
  pixel (0-15) -> re-bin to 2-bit (0-3): v = pixel // 4
  16x16 image -> 8x8 grid of 2x2 patches -> 64 integer tokens
  each token: base-4 index = v0*64 + v1*16 + v2*4 + v3  (range 0-255)

Vocabulary size: 4^4 = 256  (no UNK, every index is valid)
Sequence length: 64 tokens per image
"""

import numpy as np


VOCAB_SIZE = 256  # 4^4
SEQ_LEN = 64      # 8x8 patches
PATCH_SIZE = 2


def rebin(pixels: np.ndarray) -> np.ndarray:
    """Map 4-bit pixel values (0-15) to 2-bit bins (0-3)."""
    return pixels // 4


def encode_patch(p: np.ndarray) -> int:
    """
    Encode a 2x2 patch of 2-bit values (0-3) as a single integer 0-255.

    p: shape (2, 2) or flat (4,), dtype uint8, values 0-3
    Returns: int in [0, 255]
    """
    flat = p.ravel()
    return int(flat[0]) * 64 + int(flat[1]) * 16 + int(flat[2]) * 4 + int(flat[3])


def decode_patch(idx: int) -> np.ndarray:
    """
    Decode a patch token back to a 2x2 array of 2-bit values (0-3).

    Returns: shape (2, 2), dtype uint8, values 0-3
    """
    v3 = idx % 4;       idx //= 4
    v2 = idx % 4;       idx //= 4
    v1 = idx % 4;       idx //= 4
    v0 = idx % 4
    return np.array([[v0, v1], [v2, v3]], dtype=np.uint8)


def encode(image: np.ndarray) -> np.ndarray:
    """
    Encode a 16x16 4-bit image to a sequence of 64 patch tokens.

    image: shape (16, 16), dtype uint8, values 0-15
    Returns: shape (64,), dtype int32, values 0-255
    """
    assert image.shape == (16, 16), f"Expected (16, 16), got {image.shape}"
    binned = rebin(image)
    tokens = np.empty(SEQ_LEN, dtype=np.int32)
    for r in range(8):
        for c in range(8):
            patch = binned[r*2:(r+1)*2, c*2:(c+1)*2]
            tokens[r * 8 + c] = encode_patch(patch)
    return tokens


def decode(tokens: np.ndarray) -> np.ndarray:
    """
    Decode a sequence of 64 patch tokens back to a 16x16 image.

    tokens: shape (64,), values 0-255
    Returns: shape (16, 16), dtype uint8, values 0-3 (2-bit bins, not original 0-15)

    Note: re-binning is lossy — original 4-bit values are not recoverable.
    To display, map bins back to pixel values with: pixel = bin * 4 + 2  (bin center)
    """
    assert len(tokens) == SEQ_LEN, f"Expected {SEQ_LEN} tokens, got {len(tokens)}"
    image = np.empty((16, 16), dtype=np.uint8)
    for i, token in enumerate(tokens):
        r, c = divmod(i, 8)
        image[r*2:(r+1)*2, c*2:(c+1)*2] = decode_patch(int(token))
    return image


def bins_to_pixels(binned: np.ndarray) -> np.ndarray:
    """
    Map 2-bit bin values (0-3) back to representative pixel values (0-15).
    Uses bin center: bin 0 -> 1, bin 1 -> 5, bin 2 -> 9, bin 3 -> 13.
    """
    return binned * 4 + 1


def encode_batch(images: np.ndarray) -> np.ndarray:
    """
    Encode a batch of images.

    images: shape (N, 16, 16), dtype uint8, values 0-15
    Returns: shape (N, 64), dtype int32
    """
    N = images.shape[0]
    out = np.empty((N, SEQ_LEN), dtype=np.int32)
    for i in range(N):
        out[i] = encode(images[i])
    return out


def decode_batch(token_seqs: np.ndarray) -> np.ndarray:
    """
    Decode a batch of token sequences.

    token_seqs: shape (N, 64), values 0-255
    Returns: shape (N, 16, 16), dtype uint8, values 0-3
    """
    N = token_seqs.shape[0]
    out = np.empty((N, 16, 16), dtype=np.uint8)
    for i in range(N):
        out[i] = decode(token_seqs[i])
    return out
