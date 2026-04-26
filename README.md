# minecraft_t2i

A GPT-style transformer that generates Minecraft-style pixel art from a category name, trained on the [Quick Draw!](https://quickdraw.withgoogle.com/data) dataset. Designed to eventually run inference on a Minecraft redstone computer.

---

## How it works

1. You give it a category name (e.g. `"cat"`, `"airplane"`, `"house"`)
2. The model autoregressively generates 64 patch tokens
3. Tokens decode back to a **16×16 binary pixel art image**

Images are drawn from 345 Quick Draw categories (~50M sketches total), preprocessed to 16×16 and quantized to 4-bit. The tokenizer re-bins to binary (black/white), keeping only stroke vs. background — because Quick Draw sketches are inherently binary and intermediate gray values are resize artifacts.

---

## Architecture

| Component | Detail |
|---|---|
| Type | GPT-style autoregressive transformer |
| Parameters | ~157K |
| `d_model` | 64 |
| `n_heads` | 4 |
| `n_layers` | 4 |
| `d_ff` | 128 |
| Activation | ReLU |
| Tokenizer | 2×2 patch, binary re-bin, vocab size 16 |
| Sequence length | 65 (1 category token + 64 image tokens) |
| Positional encoding | 2D learned (row + col embeddings) |

**Tokenizer:** Each 16×16 image is re-binned to binary (`pixel // 8`), then divided into 64 non-overlapping 2×2 patches. Each patch is encoded as a 4-bit integer (0–15). Vocabulary size = 2⁴ = 16.

**Training loss:** Stroke-weighted cross-entropy — stroke patches (token ≠ 0) are weighted 10× more than background patches (token = 0). Rationale: ~70–80% of patches are background; unweighted loss lets the model ignore strokes and still achieve low average loss.

**Inference:** Greedy argmax decoding — fully deterministic, required for redstone implementation.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Dataset

### Download & preprocess

```bash
cd quickdraw_dataset
python3 download_and_preprocess.py
```

Downloads each category's `.npy` file from Google Cloud Storage, resizes 28×28 → 16×16 (LANCZOS), quantizes to 4-bit. Output saved to `quickdraw_dataset/processed/{category}.npz`.

To process only specific categories, edit `CATEGORIES_FILTER` at the top of the script.

### Curate (optional, recommended)

Filters each category to keep only the most representative images (closest to the category mean in binary space):

```bash
cd quickdraw_dataset
python3 curate.py --keep_n 5000     # keep closest 5000 per category
python3 curate.py --keep 0.3        # keep closest 30%
```

Output saved to `quickdraw_dataset/curated/`. Training on curated data produces sharper, more consistent greedy outputs.

### Visualize

```bash
# Raw processed data
python3 quickdraw_dataset/visualize.py cat dog house

# Binary tokenized view (what the model actually sees)
python3 quickdraw_dataset/visualize_binary.py cat dog
python3 quickdraw_dataset/visualize_binary.py --curated cat dog   # curated version
```

---

## Training

```bash
# Quick sanity check (~5 min)
python train.py --max_per_category 100 --epochs 3

# Small run
python train.py --max_per_category 1000 --epochs 10

# Full training on curated data (recommended)
python train.py --processed_dir quickdraw_dataset/curated --epochs 15

# Disable stroke weighting
python train.py --stroke_weight 1
```

Checkpoints are saved to `checkpoints/` after every epoch. `checkpoints/latest.pt` always points to the most recent one.

Each checkpoint contains: model weights, optimizer state, epoch number, and category list.

---

## Generation

```python
import torch
from model.transformer import PixelArtTransformer
from model.generate import generate_batch

ckpt = torch.load("checkpoints/latest.pt", map_location="cpu")
categories = ckpt["categories"]

model = PixelArtTransformer(n_categories=len(categories))
model.load_state_dict(ckpt["model"])

images = generate_batch(model, [categories.index("cat")])  # (1, 16, 16) uint8
```

See [latest_epoch_generate.py](latest_epoch_generate.py) for a full visualization script with greedy and temperature sampling.

---

## Minecraft Deployment

The goal is to run inference as a pure redstone circuit — no ComputerCraft, no mods. See [MINECRAFT.md](MINECRAFT.md) for the full implementation plan, component difficulty table, and tooling recommendations.

---

## Project structure

```
minecraft_t2i/
├── model/
│   ├── embedding.py              # category + 2D positional embeddings
│   ├── attention.py              # causal multi-head self-attention
│   ├── transformer.py            # full GPT-style decoder
│   └── generate.py               # greedy inference
├── tokenizer/
│   ├── patch_tokenizer.py        # 2×2 patch tokenizer, binary re-bin
│   └── test_patch_tokenizer.py
├── quickdraw_dataset/
│   ├── download_and_preprocess.py
│   ├── curate.py                 # filter to most representative images
│   ├── visualize.py              # raw 4-bit grid
│   └── visualize_binary.py       # original vs binary comparison
├── train.py                      # training loop
├── latest_epoch_generate.py      # visualization with greedy + temperature
├── MINECRAFT.md                  # redstone implementation plan
├── requirements.txt
└── CLAUDE.md                     # architecture notes
```
