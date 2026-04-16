"""
Download QuickDraw raw bitmap files (28x28 grayscale) from GCS,
resize to 16x16, quantize to 4-bit (values 0-15), and save per-category.

Output: quickdraw_dataset/processed/{category}.npz (compressed)
  - key: 'data'
  - dtype: uint8
  - shape: (N, 16, 16)
  - values: 0 (white/background) to 15 (black/stroke)
"""

import urllib.parse
import urllib.request
import numpy as np
from PIL import Image
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
GCS_BASE = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
CATEGORIES_URL = (
    "https://raw.githubusercontent.com/googlecreativelab/"
    "quickdraw-dataset/master/categories.txt"
)

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR / "processed"
TMP_DIR = SCRIPT_DIR / "tmp_raw"

TARGET_SIZE = (16, 16)   # output resolution
BITS = 4                 # 4-bit → 16 levels
LEVELS = 2 ** BITS       # 16

# Set to None to download all 345 categories, or a list e.g. ["cat", "dog"]
CATEGORIES_FILTER = None
# ───────────────────────────────────────────────────────────────────────────


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url) as r:
        return r.read().decode()


def download_raw(category: str, dest: Path) -> bool:
    """Download raw 28x28 GCS bitmap for a category. Returns True on success."""
    url = f"{GCS_BASE}/{urllib.parse.quote(category)}.npy"
    try:
        print(f"  downloading {url} ...", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest)
        size_mb = dest.stat().st_size / 1_048_576
        print(f"{size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"FAILED ({e})")
        return False


def preprocess(raw_path: Path, out_path: Path) -> int:
    """
    Load 28x28 uint8 images, resize to 16x16, quantize to 4-bit.
    Saves compressed .npz with key 'data'. Returns number of images processed.
    """
    data = np.load(raw_path)          # shape (N, 784), uint8, flattened 28x28
    N = len(data)
    imgs = data.reshape(N, 28, 28)    # (N, 28, 28)

    out = np.empty((N, 16, 16), dtype=np.uint8)
    for i, img in enumerate(imgs):
        pil = Image.fromarray(img, mode="L")
        pil = pil.resize(TARGET_SIZE, Image.LANCZOS)
        arr = np.asarray(pil, dtype=np.uint8)
        # Quantize: 0-255 → 0-15  (integer division by 16; 255//16 == 15)
        out[i] = arr // (256 // LEVELS)

    np.savez_compressed(out_path, data=out)
    return N


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching category list...")
    all_categories = [c.strip() for c in fetch_text(CATEGORIES_URL).splitlines() if c.strip()]
    categories = CATEGORIES_FILTER if CATEGORIES_FILTER is not None else all_categories
    print(f"  {len(categories)} categories to process\n")

    total_images = 0
    failed = []

    for idx, cat in enumerate(categories, 1):
        print(f"[{idx}/{len(categories)}] {cat}")
        out_path = OUT_DIR / f"{cat}.npz"

        if out_path.exists():
            existing = np.load(out_path)['data'].shape[0]
            print(f"  already processed ({existing} images), skipping")
            total_images += existing
            continue

        tmp_path = TMP_DIR / f"{cat}.npy"

        if not tmp_path.exists():
            ok = download_raw(cat, tmp_path)
            if not ok:
                failed.append(cat)
                continue
        else:
            print(f"  raw download already cached")

        print(f"  preprocessing to {TARGET_SIZE[0]}x{TARGET_SIZE[1]} 4-bit ...", end=" ", flush=True)
        n = preprocess(tmp_path, out_path)
        print(f"{n} images saved → {out_path.name}")
        total_images += n

        # Remove raw cache after successful processing to save disk
        tmp_path.unlink()

    print(f"\nDone. {total_images:,} total images across {len(categories) - len(failed)} categories.")
    if failed:
        print(f"Failed categories ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
