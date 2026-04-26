"""
2x2 Patch Tokenizer for 16x16 4-bit grayscale images — binary mode.

Pipeline:
  pixel (0-15) -> re-bin to binary (0-1): v = pixel // 8
  16x16 image -> 8x8 grid of 2x2 patches -> 64 integer tokens
  each token: base-2 index = v0*8 + v1*4 + v2*2 + v3  (range 0-15)

Vocabulary size: 2^4 = 16  (no UNK, every index is valid)
Sequence length: 64 tokens per image

Binary binning rationale: Quick Draw sketches are inherently binary —
black strokes on white background. Intermediate gray values in the
processed data are resize interpolation artifacts, not real signal.
"""

import numpy as np


VOCAB_SIZE = 16   # 2^4
SEQ_LEN    = 64   # 8x8 patches
PATCH_SIZE = 2


def rebin(pixels: np.ndarray) -> np.ndarray:
    """Map 4-bit pixel values (0-15) to binary bins (0-1)."""
    return pixels // 8


def encode_patch(p: np.ndarray) -> int:
    """
    Encode a 2x2 patch of binary values (0-1) as a single integer 0-15.

    p: shape (2, 2) or flat (4,), dtype uint8, values 0-1
    Returns: int in [0, 15]
    """
    flat = p.ravel()
    return int(flat[0]) * 8 + int(flat[1]) * 4 + int(flat[2]) * 2 + int(flat[3])


def decode_patch(idx: int) -> np.ndarray:
    """
    Decode a patch token back to a 2x2 array of binary values (0-1).

    Returns: shape (2, 2), dtype uint8, values 0-1
    """
    v3 = idx % 2;  idx //= 2
    v2 = idx % 2;  idx //= 2
    v1 = idx % 2;  idx //= 2
    v0 = idx % 2
    return np.array([[v0, v1], [v2, v3]], dtype=np.uint8)


def encode(image: np.ndarray) -> np.ndarray:
    """
    Encode a 16x16 4-bit image to a sequence of 64 patch tokens.

    image: shape (16, 16), dtype uint8, values 0-15
    Returns: shape (64,), dtype int32, values 0-15
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
    Decode a sequence of 64 patch tokens back to a 16x16 binary image.

    tokens: shape (64,), values 0-15
    Returns: shape (16, 16), dtype uint8, values 0-1
    """
    assert len(tokens) == SEQ_LEN, f"Expected {SEQ_LEN} tokens, got {len(tokens)}"
    image = np.empty((16, 16), dtype=np.uint8)
    for i, token in enumerate(tokens):
        r, c = divmod(i, 8)
        image[r*2:(r+1)*2, c*2:(c+1)*2] = decode_patch(int(token))
    return image


def bins_to_pixels(binned: np.ndarray) -> np.ndarray:
    """
    Map binary bin values (0-1) to pixel values for display.
      bin 0 (background) -> 0
      bin 1 (stroke)     -> 15
    Use  15 - img  in imshow to get white background / black strokes.
    """
    return (binned * 15).astype(np.uint8)


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

    token_seqs: shape (N, 64), values 0-15
    Returns: shape (N, 16, 16), dtype uint8, values 0-1
    """
    N = token_seqs.shape[0]
    out = np.empty((N, 16, 16), dtype=np.uint8)
    for i in range(N):
        out[i] = decode(token_seqs[i])
    return out
