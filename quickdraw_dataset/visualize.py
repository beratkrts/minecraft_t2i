"""
Visualize processed 16x16 4-bit QuickDraw images.

Usage:
    python3 visualize.py                    # random sample from all processed files
    python3 visualize.py cat dog house      # specific categories
    python3 visualize.py --n 12             # 12 samples per category (default: 8)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROCESSED_DIR = SCRIPT_DIR / "processed"


def load_category(category: str) -> np.ndarray | None:
    path = PROCESSED_DIR / f"{category}.npz"
    if not path.exists():
        print(f"  '{category}' not found in processed/")
        return None
    return np.load(path)['data']


def pick_categories(requested: list[str]) -> list[str]:
    if requested:
        return requested
    existing = sorted(PROCESSED_DIR.glob("*.npz"))
    if not existing:
        print("No processed files found. Run download_and_preprocess.py first.")
        return []
    return [p.stem for p in existing[:12]]  # up to 12 for readability


def render(categories: list[str], n_samples: int, save_path: Path):
    n_cats = len(categories)
    fig = plt.figure(figsize=(max(n_samples * 1.2, 10), n_cats * 1.4 + 0.6), facecolor="#1a1a2e")
    fig.suptitle(
        "QuickDraw — 16×16 · 4-bit grayscale",
        color="white", fontsize=13, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(
        n_cats, n_samples, figure=fig,
        hspace=0.55, wspace=0.08,
        left=0.08, right=0.99, top=0.93, bottom=0.02,
    )

    rng = np.random.default_rng(42)

    for row, cat in enumerate(categories):
        data = load_category(cat)
        if data is None:
            continue

        indices = rng.choice(len(data), size=min(n_samples, len(data)), replace=False)

        for col, idx in enumerate(indices):
            ax = fig.add_subplot(gs[row, col])
            img = (data[idx].astype(np.float32) / 15.0 * 255).astype(np.uint8)
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")
                spine.set_linewidth(0.5)
            if col == 0:
                ax.set_ylabel(cat, color="#aaaadd", fontsize=8,
                              rotation=0, labelpad=38, va="center")

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize 16x16 4-bit QuickDraw images")
    parser.add_argument("categories", nargs="*", help="Category names (e.g. cat dog house)")
    parser.add_argument("--n", type=int, default=8, dest="n_samples",
                        help="Samples per category (default: 8)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save output PNG path (default: preview.png)")
    args = parser.parse_args()

    categories = pick_categories(args.categories)
    if not categories:
        return

    print(f"Showing {args.n_samples} samples × {len(categories)} categories\n")
    save_path = Path(args.save) if args.save else SCRIPT_DIR / "preview.png"
    render(categories, args.n_samples, save_path)


if __name__ == "__main__":
    main()
