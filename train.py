"""
Training script for PixelArtTransformer.

Usage:
  python train.py
  python train.py --epochs 20 --batch_size 128 --lr 3e-4
  python train.py --max_per_category 5000 --checkpoint_dir checkpoints/run1

Data:
  Reads all .npz files from quickdraw_dataset/processed/.
  Each file is one Quick Draw category; categories are sorted alphabetically
  and assigned indices 0..N-1. The same sorted list is saved alongside
  checkpoints so inference can recover the mapping.

Checkpoints:
  Saved after every epoch as checkpoint_dir/epoch_{N:03d}.pt
  Each checkpoint contains: model state, optimizer state, epoch, category list.
  Latest checkpoint is also copied to checkpoint_dir/latest.pt.
"""

import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model.transformer import PixelArtTransformer
from tokenizer.patch_tokenizer import encode_batch, SEQ_LEN


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class QuickDrawDataset(Dataset):
    """
    Loads all processed Quick Draw .npz files, encodes images with the
    2x2 patch tokenizer, and stores everything as uint8 tensors in memory.

    Args:
        processed_dir     : path to quickdraw_dataset/processed/
        max_per_category  : cap images per category (None = use all)
    """

    def __init__(self, processed_dir: str, max_per_category: int | None = None):
        npz_files = sorted(
            f for f in os.listdir(processed_dir) if f.endswith(".npz")
        )
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {processed_dir}")

        self.categories = [os.path.splitext(f)[0] for f in npz_files]

        all_tokens = []
        all_labels = []

        print(f"Loading {len(npz_files)} categories from {processed_dir} ...")
        for cat_idx, fname in enumerate(npz_files):
            path = os.path.join(processed_dir, fname)
            images = np.load(path)["data"]          # (N, 16, 16) uint8 0-15

            if max_per_category is not None:
                images = images[:max_per_category]

            tokens = encode_batch(images)           # (N, 64) int32 0-255
            all_tokens.append(tokens.astype(np.uint8))
            all_labels.append(np.full(len(tokens), cat_idx, dtype=np.int16))

        self.tokens = torch.from_numpy(
            np.concatenate(all_tokens, axis=0)      # (total, 64) uint8
        ).long()
        self.labels = torch.from_numpy(
            np.concatenate(all_labels, axis=0)      # (total,) int16
        ).long()

        print(f"Dataset ready: {len(self.tokens):,} images across {len(self.categories)} categories")

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.labels[idx], self.tokens[idx]   # (scalar, 64)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = (
        "cuda"  if torch.cuda.is_available()  else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # Data
    dataset = QuickDrawDataset(
        processed_dir=args.processed_dir,
        max_per_category=args.max_per_category,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device != "cpu"),
    )

    # Model
    model = PixelArtTransformer(
        n_categories=len(dataset.categories),
        vocab_size=16,
        d_model=64,
        n_heads=4,
        n_layers=4,
        d_ff=128,
    ).to(device)
    print(f"Parameters: {model.num_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )

    # NOTE: Stroke-weighted loss — stroke patches (token != 0) are weighted
    # args.stroke_weight x more than background patches (token == 0).
    # Rationale: ~70-80% of patches are background; unweighted loss lets the
    # model ignore stroke positions and still achieve low average loss.
    # To disable, set --stroke_weight 1.
    stroke_weight = args.stroke_weight

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Save category list so inference can recover the index→name mapping
    cat_list_path = os.path.join(args.checkpoint_dir, "categories.txt")
    with open(cat_list_path, "w") as f:
        f.write("\n".join(dataset.categories))

    # Training
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for step, (cat_idx, tokens) in enumerate(loader):
            cat_idx = cat_idx.to(device)       # (B,)
            tokens  = tokens.to(device)        # (B, 64)

            # Forward: feed all 64 tokens; logits[:, :64] predicts tokens[:, 0:64]
            logits = model(cat_idx, tokens)    # (B, 65, 256)
            logits = logits[:, :SEQ_LEN, :]    # (B, 64, 256) — drop last unused position

            # Loss: stroke-weighted cross-entropy
            # background patch (token=0) → weight 1, stroke patch → weight stroke_weight
            flat_logits = logits.reshape(-1, model.vocab_size)  # (B*64, 16)
            flat_tokens = tokens.reshape(-1)                    # (B*64,)
            per_token_loss = F.cross_entropy(flat_logits, flat_tokens, reduction='none')
            weights = torch.where(flat_tokens == 0, 1.0, float(stroke_weight))
            loss = (per_token_loss * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

            if (step + 1) % args.log_every == 0:
                avg = total_loss / n_batches
                print(f"  epoch {epoch:3d}  step {step+1:6d}  loss {avg:.4f}")

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch:3d} complete — avg loss {avg_loss:.4f}")

        # Checkpoint
        ckpt = {
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "categories": dataset.categories,
        }
        epoch_path  = os.path.join(args.checkpoint_dir, f"epoch_{epoch:03d}.pt")
        latest_path = os.path.join(args.checkpoint_dir, "latest.pt")
        torch.save(ckpt, epoch_path)
        shutil.copy(epoch_path, latest_path)
        print(f"Checkpoint saved → {epoch_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train PixelArtTransformer")
    p.add_argument("--processed_dir",    default="quickdraw_dataset/processed")
    p.add_argument("--checkpoint_dir",   default="checkpoints")
    p.add_argument("--epochs",           type=int,   default=10)
    p.add_argument("--batch_size",       type=int,   default=128)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--max_per_category", type=int,   default=None,
                   help="Cap images per category. Omit to use all data.")
    p.add_argument("--log_every",        type=int,   default=100)
    p.add_argument("--num_workers",      type=int,   default=4)
    p.add_argument("--stroke_weight",    type=float, default=10.0,
                   help="Loss weight for stroke patches (token != 0). Set to 1 to disable.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
