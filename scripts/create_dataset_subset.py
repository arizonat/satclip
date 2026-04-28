#!/usr/bin/env python3
"""
Uniformly sample 10K images per year from dataset/images/<year>/<images.tif>
into dataset_subset/images/<year>/ with the same folder structure.
"""

import os
import shutil
import random
from pathlib import Path
from math import ceil
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
SRC_ROOT = Path("dataset/images")
DST_ROOT = Path("dataset_subset/images")
N_SAMPLES = 10_000
SEED      = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)

year_dirs = sorted([d for d in SRC_ROOT.iterdir() if d.is_dir()])

for year_dir in tqdm(year_dirs, desc="Processing"):
    images = list(year_dir.glob("*.tif"))
    n = len(images)

    if n == 0:
        print(f"[{year_dir.name}] No .tif files found, skipping.")
        continue

    if n <= N_SAMPLES:
        # Take all images if fewer than N_SAMPLES exist
        sampled = images
        print(f"[{year_dir.name}] Only {n} images available — copying all.")
    else:
        # Uniform stride sample, then shuffle to avoid bias
        stride   = n / N_SAMPLES
        sampled  = [images[int(i * stride)] for i in range(N_SAMPLES)]
        print(f"[{year_dir.name}] {n} images → sampling {N_SAMPLES} (stride={stride:.2f})")

    dst_dir = DST_ROOT / year_dir.name
    dst_dir.mkdir(parents=True, exist_ok=True)

    for src in sampled:
        shutil.copy2(src, dst_dir / src.name)

print("\nDone.")