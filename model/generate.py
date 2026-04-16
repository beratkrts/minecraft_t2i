"""
Greedy autoregressive inference for PixelArtTransformer.

Generates one image at a time (or a batch) by repeatedly:
  1. Running the full sequence through the transformer
  2. Taking argmax of the last logit position
  3. Appending the predicted token

Greedy decoding is deterministic — required for Minecraft redstone deployment.
"""

import torch
import numpy as np

from tokenizer.patch_tokenizer import decode as tokens_to_image, bins_to_pixels, SEQ_LEN


@torch.no_grad()
def generate(
    model,
    cat_idx: int,
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate a single 16x16 image for the given category index.

    Returns: (16, 16) uint8 array with pixel values 1,5,9,13 (bin centres × 4 + 1)
    """
    images = generate_batch(model, [cat_idx], device=device)
    return images[0]


@torch.no_grad()
def generate_batch(
    model,
    cat_indices,          # list or 1-D array of category indices
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate one image per category index.

    cat_indices : sequence of ints, length B
    Returns: (B, 16, 16) uint8 array with pixel values (bin centres)
    """
    model.eval()
    B = len(cat_indices)

    cat = torch.tensor(cat_indices, dtype=torch.long, device=device)  # (B,)
    # Start with empty token sequence
    tokens = torch.empty((B, 0), dtype=torch.long, device=device)

    for _ in range(SEQ_LEN):
        logits = model(cat, tokens)          # (B, T+1, vocab_size)
        next_token = logits[:, -1, :].argmax(dim=-1)   # (B,)
        tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)  # (B, T+1)

    # Decode tokens → 2-bit binned images → display pixels
    token_np = tokens.cpu().numpy()                 # (B, 64)
    images = np.empty((B, 16, 16), dtype=np.uint8)
    for i in range(B):
        binned = tokens_to_image(token_np[i])        # (16, 16) values 0-3
        images[i] = bins_to_pixels(binned)           # map to 1,5,9,13
    return images
