# CLAUDE.md

## Project overview

Transformer model for text-to-image generation of Minecraft-style pixel art. Trained on Quick Draw drawings preprocessed to 16×16 4-bit grayscale (values 0–15, 16 gray levels).

## Environment

- Python 3.11, venv at `.venv/`
- Install deps: `pip install -r requirements.txt`

## Data pipeline

- Source: [Quick Draw Dataset](https://github.com/googlecreativelab/quickdraw-dataset) — 345 categories, ~50M drawings
- Raw format: 28×28 uint8 grayscale `.npy` files from GCS
- Processed format: `(N, 16, 16)` uint8, values 0–15 (4-bit), saved per category in `quickdraw_dataset/processed/` as compressed `.npz` files (key: `'data'`)
- Scripts:
  - `quickdraw_dataset/download_and_preprocess.py` — download + resize + quantize
  - `quickdraw_dataset/visualize.py` — grid visualization of processed images, reads only from `processed/`

## Image format

- Resolution: 16×16 pixels
- Bit depth: 4-bit (16 levels)
- Encoding: `pixel_value = raw_28x28_pixel // 16`, so 0 = white/background, 15 = black/stroke
- Storage: compressed `.npz` per category, accessed as `np.load(path)['data']`; values 0–15, not nibble-packed

## Conventions

- Do not add download/fetch logic to `visualize.py` — that belongs in `download_and_preprocess.py`
- `processed/` and `tmp_npy/` directories are data-only and should be gitignored

---

## Model Architecture

### Task
Autoregressive text-to-image generation: given a category name (one of 345 Quick Draw classes), generate a 16×16 4-bit grayscale image token-by-token.

### Tokenizer

Two tokenizer variants are implemented as swappable modules under `tokenizer/`.

#### Option A: 2×2 Patch Tokenizer (primary)
1. **Re-bin** each pixel from 4-bit (0–15) to binary (0–1): `v = pixel // 8`
2. **Patch** the 16×16 image into an 8×8 grid of non-overlapping 2×2 blocks → 64 patches
3. **Encode** each 2×2 patch as a single integer index (base-2, 4 digits): `idx = v0*8 + v1*4 + v2*2 + v3`
4. **Vocabulary size**: 2^4 = **16** (all indices 0–15 are valid; no UNK needed)
5. **Sequence length**: 64 tokens per image
6. Scan order: raster (left→right, top→bottom)

Rationale: Quick Draw sketches are inherently binary — intermediate gray values are resize artifacts, not real signal. Binary binning gives a 16-way classification problem vs 256-way, making greedy decoding much more reliable.

Inverse: `idx → (v0,v1,v2,v3)` via base-2 decomposition → `pixel = v * 15` → 16×16 image

#### Option B: Byte Pair Encoding (secondary, for comparison)
- Flatten 16×16 = 256 pixel values (0–15 alphabet, 16 initial tokens)
- BPE merges ~240 rules → vocab_size = 256, variable-length sequences (~100–180 tokens)
- Harder to implement in Minecraft inference; used for perplexity comparison only

### Text Conditioning
- Input: category index (integer 0–344)
- **Learned category embedding**: lookup table `(345, d_model)`
- Prepended as token 0; no cross-attention needed

### Transformer (GPT-style autoregressive decoder)

| Hyperparameter | Value |
|---|---|
| `d_model` | 64 |
| `n_heads` | 4 (head_dim = 16) |
| `n_layers` | 4 |
| `d_ff` | 128 (2× d_model) |
| `vocab_size` | 16 |
| `max_seq_len` | 65 (1 category + 64 image tokens) |
| Activation | ReLU |
| Positional encoding | 2D learned: `row_embed(8, d_model) + col_embed(8, d_model)` |
| Attention mask | Causal (lower-triangular) |

Estimated parameters: ~260K → ~260KB at int8 quantization.

### Positional Encoding
- Token 0 (category): no position embedding applied
- Tokens 1–64 (image patches): 2D learned position = `row_embed[r] + col_embed[c]` where `r, c ∈ {0,...,7}`
- Parameters: 2 × 8 × 64 = 1,024 (vs. 64 × 64 = 4,096 for flat 1D)

### Training
- Objective: autoregressive cross-entropy on image tokens only (token 0 is input, not predicted)
- Teacher forcing during training
- Optimizer: AdamW
- No dropout at inference

### Inference (Minecraft deployment)
1. Category index → embedding lookup
2. Autoregressively sample 64 tokens via **greedy argmax** (deterministic, no sampling)
3. Map 64 patch tokens → 8×8 grid → each 2×2 block → 16×16 image
4. Post-training: export int8 quantized weights for redstone implementation

### File structure
```
tokenizer/
  patch_tokenizer.py   # 2×2 patch tokenizer (primary)
  bpe_tokenizer.py     # BPE tokenizer (secondary)
model/
  embedding.py         # category + positional embeddings
  attention.py         # causal multi-head self-attention
  transformer.py       # full GPT-style decoder
  generate.py          # greedy inference
train.py               # training loop
export.py              # int8 weight export for Minecraft
```
