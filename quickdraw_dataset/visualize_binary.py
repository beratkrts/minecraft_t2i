"""
Visualize QuickDraw images after binary tokenization.

Shows what the model actually sees during training:
  raw 16x16 4-bit  →  re-bin (pixel // 8)  →  binary 16x16 (0 or 1)

Left column : original 4-bit grayscale
Right column: binary version (what the tokenizer produces)

Usage:
    python3 visualize_binary.py                        # random sample from all categories
    python3 visualize_binary.py cat dog airplane       # specific categories
    python3 visualize_binary.py --n 6                  # 6 samples per category (default: 5)
    python3 visualize_binary.py --curated              # use curated/ instead of processed/
    python3 visualize_binary.py cat dog --curated --n 6
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

SCRIPT_DIR    = Path(__file__).parent
PROCESSED_DIR = SCRIPT_DIR / "processed"
CURATED_DIR   = SCRIPT_DIR / "curated"
sys.path.insert(0, str(SCRIPT_DIR.parent))

from tokenizer.patch_tokenizer import rebin, bins_to_pixels


def load_category(category: str, data_dir: Path) -> np.ndarray | None:
    path = data_dir / f"{category}.npz"
    if not path.exists():
        print(f"  '{category}' not found in {data_dir.name}/")
        return None
    return np.load(path)["data"]


def pick_categories(requested: list[str], data_dir: Path) -> list[str]:
    if requested:
        return requested
    existing = sorted(data_dir.glob("*.npz"))
    if not existing:
        print(f"No .npz files found in {data_dir}/")
        return []
    return [p.stem for p in existing[:8]]


def render(categories: list[str], n_samples: int, save_path: Path, data_dir: Path):
    n_cats = len(categories)
    # Two columns per sample: original | binary
    n_cols = n_samples * 2

    fig = plt.figure(
        figsize=(max(n_cols * 1.1, 10), n_cats * 1.6 + 0.8),
        facecolor="#1a1a2e",
    )
    source_label = "curated" if data_dir == CURATED_DIR else "processed"
    fig.suptitle(
        f"QuickDraw [{source_label}] — original 4-bit (left)  vs  binary tokenized (right)",
        color="white", fontsize=12, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(
        n_cats, n_cols, figure=fig,
        hspace=0.5, wspace=0.05,
        left=0.1, right=0.99, top=0.93, bottom=0.02,
    )

    rng = np.random.default_rng(42)

    for row, cat in enumerate(categories):
        data = load_category(cat, data_dir)
        if data is None:
            continue

        indices = rng.choice(len(data), size=min(n_samples, len(data)), replace=False)

        for i, idx in enumerate(indices):
            original = data[idx]                         # (16,16) uint8, 0-15
            binary   = bins_to_pixels(rebin(original))   # (16,16) uint8, 0 or 15

            # Original
            ax_orig = fig.add_subplot(gs[row, i * 2])
            ax_orig.imshow(15 - original, cmap="gray", vmin=0, vmax=15,
                           interpolation="nearest")
            ax_orig.set_xticks([])
            ax_orig.set_yticks([])
            for spine in ax_orig.spines.values():
                spine.set_edgecolor("#555577")
                spine.set_linewidth(0.5)

            # Binary
            ax_bin = fig.add_subplot(gs[row, i * 2 + 1])
            ax_bin.imshow(15 - binary, cmap="gray", vmin=0, vmax=15,
                          interpolation="nearest")
            ax_bin.set_xticks([])
            ax_bin.set_yticks([])
            for spine in ax_bin.spines.values():
                spine.set_edgecolor("#7777aa")
                spine.set_linewidth(0.8)

            if i == 0:
                ax_orig.set_ylabel(cat, color="#aaaadd", fontsize=8,
                                   rotation=0, labelpad=42, va="center")

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize binary-tokenized QuickDraw images")
    parser.add_argument("categories", nargs="*")
    parser.add_argument("--n", type=int, default=5, dest="n_samples")
    parser.add_argument("--curated", action="store_true",
                        help="Use curated/ instead of processed/")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    data_dir = CURATED_DIR if args.curated else PROCESSED_DIR

    if args.curated and not data_dir.exists():
        print("curated/ not found. Run curate.py first.")
        return

    categories = pick_categories(args.categories, data_dir)
    if not categories:
        return

    source = "curated" if args.curated else "processed"
    print(f"Source: {source}/")
    print(f"Showing {args.n_samples} samples × {len(categories)} categories\n")
    save_path = Path(args.save) if args.save else SCRIPT_DIR / "preview_binary.png"
    render(categories, args.n_samples, save_path, data_dir)


if __name__ == "__main__":
    main()
