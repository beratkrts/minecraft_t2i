"""Quick sanity checks for the binary 2x2 patch tokenizer."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tokenizer.patch_tokenizer import (
    encode, decode, encode_patch, decode_patch,
    rebin, bins_to_pixels, encode_batch, decode_batch,
    VOCAB_SIZE, SEQ_LEN,
)


def test_vocab_size():
    assert VOCAB_SIZE == 16
    assert SEQ_LEN == 64


def test_rebin():
    pixels = np.array([0, 7, 8, 15], dtype=np.uint8)
    expected = np.array([0, 0, 1, 1], dtype=np.uint8)
    assert np.array_equal(rebin(pixels), expected)


def test_patch_encode_decode_roundtrip():
    # All-zero patch (all background)
    p = np.zeros((2, 2), dtype=np.uint8)
    assert encode_patch(p) == 0
    assert np.array_equal(decode_patch(0), p)

    # All-ones patch (all stroke)
    p = np.full((2, 2), 1, dtype=np.uint8)
    assert encode_patch(p) == 15
    assert np.array_equal(decode_patch(15), p)

    # Mixed patch
    p = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    idx = encode_patch(p)
    assert 0 <= idx < 16
    assert np.array_equal(decode_patch(idx), p)


def test_all_patch_indices_covered():
    """Every integer 0-15 must decode to a valid patch and re-encode to the same index."""
    for idx in range(16):
        patch = decode_patch(idx)
        assert patch.shape == (2, 2)
        assert patch.dtype == np.uint8
        assert np.all((patch == 0) | (patch == 1))
        assert encode_patch(patch) == idx


def test_image_encode_decode_roundtrip():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 16, size=(16, 16), dtype=np.uint8)

    tokens = encode(image)
    assert tokens.shape == (64,)
    assert tokens.dtype == np.int32
    assert np.all(tokens >= 0) and np.all(tokens < 16)

    reconstructed = decode(tokens)
    assert reconstructed.shape == (16, 16)
    assert np.array_equal(reconstructed, rebin(image))


def test_all_white_image():
    image = np.zeros((16, 16), dtype=np.uint8)
    tokens = encode(image)
    assert np.all(tokens == 0)
    assert np.array_equal(decode(tokens), np.zeros((16, 16), dtype=np.uint8))


def test_all_black_image():
    image = np.full((16, 16), 15, dtype=np.uint8)
    tokens = encode(image)
    assert np.all(tokens == 15)
    assert np.array_equal(decode(tokens), np.ones((16, 16), dtype=np.uint8))


def test_bins_to_pixels():
    binned = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    pixels = bins_to_pixels(binned)
    assert pixels[0, 0] == 0
    assert pixels[0, 1] == 15
    assert pixels[1, 0] == 15
    assert pixels[1, 1] == 0


def test_batch_encode_decode():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 16, size=(10, 16, 16), dtype=np.uint8)
    tokens = encode_batch(images)
    assert tokens.shape == (10, 64)
    reconstructed = decode_batch(tokens)
    assert reconstructed.shape == (10, 16, 16)
    assert np.array_equal(reconstructed, rebin(images))


def test_token_order_is_raster():
    image = np.zeros((16, 16), dtype=np.uint8)
    image[0:2, 0:2] = 15   # top-left patch all stroke
    tokens = encode(image)
    assert tokens[0] == 15  # all-ones patch = 1*8+1*4+1*2+1 = 15
    assert np.all(tokens[1:] == 0)


if __name__ == '__main__':
    tests = [
        test_vocab_size,
        test_rebin,
        test_patch_encode_decode_roundtrip,
        test_all_patch_indices_covered,
        test_image_encode_decode_roundtrip,
        test_all_white_image,
        test_all_black_image,
        test_bins_to_pixels,
        test_batch_encode_decode,
        test_token_order_is_raster,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
