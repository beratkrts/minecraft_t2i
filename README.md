# minecraft_t2i

A GPT-style transformer that generates Minecraft-style pixel art from a category name, trained on the [Quick Draw!](https://quickdraw.withgoogle.com/data) dataset. Designed to run inference on a Minecraft redstone computer.

---

## How it works

1. You give it a category name (e.g. `"cat"`, `"airplane"`, `"house"`)
2. The model autoregressively generates 64 patch tokens
3. Tokens decode back to a **16×16 4-bit grayscale image**

Images are drawn from 345 Quick Draw categories (~50M sketches total), preprocessed to 16×16 and quantized to 4-bit (values 0–15).

---

## Architecture

| Component | Detail |
|---|---|
| Type | GPT-style autoregressive transformer |
| Parameters | ~188K |
| `d_model` | 64 |
| `n_heads` | 4 |
| `n_layers` | 4 |
| `d_ff` | 128 |
| Activation | ReLU |
| Tokenizer | 2×2 patch, vocab size 256 |
| Sequence length | 65 (1 category token + 64 image tokens) |
| Positional encoding | 2D learned (row + col embeddings) |

**Tokenizer:** Each 16×16 image is divided into 64 non-overlapping 2×2 patches. Pixel values are re-binned from 4-bit (0–15) to 2-bit (0–3), giving 4⁴ = 256 possible patch tokens. No UNK token needed.

**Inference:** Greedy argmax decoding — fully deterministic, suitable for redstone implementation.

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

To process only specific categories, edit `CATEGORIES_FILTER` at the top of the script:

```python
CATEGORIES_FILTER = ["cat", "dog", "house"]
```

### Visualize raw data

```bash
python3 quickdraw_dataset/visualize.py                  # all categories
python3 quickdraw_dataset/visualize.py cat dog house    # specific categories
python3 quickdraw_dataset/visualize.py --n 12           # 12 samples per row
```

---

## Training

```bash
# Quick sanity check (~5 min)
python train.py --max_per_category 100 --epochs 3

# Small run
python train.py --max_per_category 1000 --epochs 10

# Full training
python train.py --epochs 20
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

## Project structure

```
minecraft_t2i/
├── model/
│   ├── embedding.py         # category + 2D positional embeddings
│   ├── attention.py         # causal multi-head self-attention
│   ├── transformer.py       # full GPT-style decoder
│   └── generate.py          # greedy inference
├── tokenizer/
│   ├── patch_tokenizer.py   # 2x2 patch tokenizer (primary)
│   └── test_patch_tokenizer.py
├── quickdraw_dataset/
│   ├── download_and_preprocess.py
│   └── visualize.py
├── train.py                 # training loop
├── latest_epoch_generate.py # visualization script
├── requirements.txt
└── CLAUDE.md                # architecture notes
```
