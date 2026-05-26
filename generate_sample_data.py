"""
Generate synthetic sample data for testing the app before real data arrives.

Creates:
  data/embeddings.npy   — 500 patches × 512-dim float32
  data/metadata.csv     — matching metadata
  data/patches/         — 500 synthetic 256×256 PNG images

Usage:
    python generate_sample_data.py
    python generate_sample_data.py --n 2000   # larger test set
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

DATA_DIR = Path("data")
PATCHES_DIR = DATA_DIR / "patches"
DATA_DIR.mkdir(exist_ok=True)
PATCHES_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)


def make_synthetic_patch(idx: int, cluster: int, n_clusters: int) -> Image.Image:
    """
    Generate a 256×256 synthetic brain-scan-like image.
    Different clusters have different patterns:
      - dense signal clusters: bright blobs on dark background
      - sparse signal clusters: few sparse dots
      - texture clusters: gradient + noise
    """
    img = Image.new("RGB", (256, 256), color=(8, 8, 12))
    draw = ImageDraw.Draw(img)

    category = cluster % 3
    palette_hue = int(cluster * 255 / n_clusters)

    if category == 0:
        # Dense fluorescent blobs
        for _ in range(rng.integers(10, 30)):
            x, y = rng.integers(10, 246, size=2)
            r = rng.integers(3, 15)
            brightness = rng.integers(150, 255)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(brightness, brightness // 2, 40))
        img = img.filter(ImageFilter.GaussianBlur(1.5))

    elif category == 1:
        # Sparse single neurons
        for _ in range(rng.integers(2, 8)):
            x, y = rng.integers(20, 236, size=2)
            r = rng.integers(4, 10)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(220, 200, 30))
        # Noise floor
        noise = rng.integers(0, 30, size=(256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(np.clip(np.array(img) + noise, 0, 255).astype(np.uint8))

    else:
        # Gradient + texture (low signal)
        arr = np.zeros((256, 256, 3), dtype=np.float32)
        x_grad = np.linspace(0, 40, 256)
        arr[:, :, 0] = x_grad[np.newaxis, :]
        noise = rng.normal(0, 15, (256, 256, 3))
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def image_stats(img: Image.Image) -> dict:
    arr = np.array(img, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    # Laplacian variance as sharpness proxy
    from PIL import ImageFilter
    lap = np.array(img.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(lap.var() / (255 ** 2)),
        "snr": float(gray.mean() / (gray.std() + 1e-6)),
    }


def main(n: int = 500):
    print(f"Generating {n} synthetic patches…")
    n_clusters = max(8, n // 50)

    # Embeddings: structured clusters in 512-dim space
    embeddings = []
    rows = []
    brain_ids = [f"brain_{i:02d}" for i in range(1, 13)]
    conditions = ["placebo"] * 6 + ["semaglutide"] * 6

    for i in range(n):
        cl = int(i * n_clusters / n)
        # Cluster centre + noise
        centre = rng.standard_normal(512) * 3
        vec = centre + rng.standard_normal(512) * 0.5
        embeddings.append(vec.astype(np.float32))

        brain_idx = i % 12
        brain_id = brain_ids[brain_idx]
        condition = conditions[brain_idx]

        patch_id = f"patch_{i:05d}"
        img = make_synthetic_patch(i, cl, n_clusters)
        img.save(PATCHES_DIR / f"{patch_id}.png")

        stats = image_stats(img)
        rows.append({
            "patch_id": patch_id,
            "brain_id": brain_id,
            "condition": condition,
            "cluster_true": cl,  # ground truth for verification
            **stats,
        })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n} patches generated")

    emb_arr = np.stack(embeddings)
    np.save(DATA_DIR / "embeddings.npy", emb_arr)
    print(f"Saved embeddings: {emb_arr.shape}")

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "metadata.csv", index=False)
    print(f"Saved metadata: {len(df)} rows")
    print("\nDone! Run: streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="Number of patches to generate")
    args = parser.parse_args()
    main(args.n)
