"""
Dataset curation: keep only the most representative images per category.

Strategy:
  1. Re-bin each image to binary (pixel // 8) — same as the tokenizer
  2. Compute the mean binary image for the category
  3. Compute L2 distance of every image from the mean
  4. Keep the N closest images (--keep_n) or closest fraction (--keep)

Output saved to quickdraw_dataset/curated/ with same format as processed/
(.npz, key 'data', shape (N, 16, 16), uint8, values 0-15 original scale).

Usage:
    python3 curate.py --keep_n 5000          # keep closest 5000 per category
    python3 curate.py --keep 0.3             # keep closest 30% per category
    python3 curate.py cat dog --keep_n 5000  # specific categories only
"""

import argparse
import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR    = Path(__file__).parent
PROCESSED_DIR = SCRIPT_DIR / "processed"
CURATED_DIR   = SCRIPT_DIR / "curated"

sys.path.insert(0, str(SCRIPT_DIR.parent))
from tokenizer.patch_tokenizer import rebin


def curate_category(images: np.ndarray, n_keep: int) -> np.ndarray:
    """
    images : (N, 16, 16) uint8, values 0-15
    n_keep : how many to keep

    Returns filtered images in original 0-15 scale, sorted closest-first.
    """
    binned = rebin(images).astype(np.float32)   # (N, 16, 16) values 0-1
    mean   = binned.mean(axis=0)                # (16, 16)
    diff   = binned - mean
    dist   = (diff ** 2).sum(axis=(1, 2))       # (N,)
    closest = np.argsort(dist)[:n_keep]
    return images[closest]


def main():
    parser = argparse.ArgumentParser(description="Curate QuickDraw dataset")
    parser.add_argument("categories", nargs="*",
                        help="Category names (default: all)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--keep_n", type=int, default=None,
                       help="Keep closest N images per category (e.g. --keep_n 5000)")
    group.add_argument("--keep", type=float, default=None,
                       help="Keep closest fraction per category (e.g. --keep 0.3)")

    args = parser.parse_args()

    # Default: keep_n 5000
    if args.keep_n is None and args.keep is None:
        args.keep_n = 5000

    if args.keep is not None and not (0 < args.keep <= 1.0):
        print("--keep must be between 0 and 1")
        sys.exit(1)

    # Collect categories
    if args.categories:
        npz_files = [PROCESSED_DIR / f"{c}.npz" for c in args.categories]
        missing = [f for f in npz_files if not f.exists()]
        if missing:
            print(f"Not found: {[f.stem for f in missing]}")
            sys.exit(1)
    else:
        npz_files = sorted(PROCESSED_DIR.glob("*.npz"))

    if not npz_files:
        print("No .npz files found in processed/. Run download_and_preprocess.py first.")
        sys.exit(1)

    CURATED_DIR.mkdir(exist_ok=True)

    if args.keep_n:
        print(f"Curating {len(npz_files)} categories — keeping closest {args.keep_n} per category")
    else:
        print(f"Curating {len(npz_files)} categories — keeping closest {args.keep*100:.0f}%")
    print(f"Output → {CURATED_DIR}\n")

    total_before = 0
    total_after  = 0

    for npz_path in npz_files:
        category = npz_path.stem
        images   = np.load(npz_path)["data"]    # (N, 16, 16) uint8
        n_before = len(images)

        if args.keep_n:
            n_keep = min(args.keep_n, n_before)
        else:
            n_keep = max(1, int(n_before * args.keep))

        curated = curate_category(images, n_keep)
        n_after = len(curated)

        np.savez_compressed(CURATED_DIR / f"{category}.npz", data=curated)

        total_before += n_before
        total_after  += n_after
        print(f"  {category:<35} {n_before:>7} → {n_after:>6}")

    print(f"\nDone. {total_before:,} → {total_after:,} images")
    print(f"Saved to {CURATED_DIR}/")
    print(f"\nTo train on curated data:")
    print(f"  python train.py --processed_dir quickdraw_dataset/curated --epochs 15")


if __name__ == "__main__":
    main()
