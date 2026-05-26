"""
Data loading utilities.

Expected data layout (put files in data/):
  embeddings.npy   — float32 array of shape (N, 512)
  metadata.csv     — one row per patch with columns:
                       patch_id, brain_id, condition,
                       brightness, sharpness, contrast, snr
  patches/         — 256×256 images named <patch_id>.jpg (or .png)
                     OR a single patches.h5 with dataset "images" shape (N,256,256,3)
                     keyed by patch_id stored in dataset "patch_ids".

The loader is forgiving: missing optional columns are filled with defaults.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from PIL import Image

DATA_DIR = Path(__file__).parent.parent / "data"
PATCHES_DIR = DATA_DIR / "patches"
H5_PATH = DATA_DIR / "patches.h5"


# ── embeddings ──────────────────────────────────────────────────────────────

def load_embeddings(path: Optional[str | Path] = None) -> np.ndarray:
    """Load (N, D) float32 embeddings array."""
    if path is None:
        path = DATA_DIR / "embeddings.npy"
    arr = np.load(path).astype(np.float32)
    assert arr.ndim == 2, f"Expected 2D array, got shape {arr.shape}"
    return arr


# ── metadata ─────────────────────────────────────────────────────────────────

_REQUIRED_COLS = {"patch_id"}
_OPTIONAL_DEFAULTS = {
    "brain_id": "unknown",
    "condition": "unknown",
    "brightness": 0.5,
    "sharpness": 0.5,
    "contrast": 0.5,
    "snr": 0.5,
}


def load_metadata(path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load per-patch metadata CSV.
    Adds a derived `quality_score` column = mean(sharpness, snr).
    Missing optional columns are filled with defaults.
    """
    if path is None:
        path = DATA_DIR / "metadata.csv"

    df = pd.read_csv(path)

    # Ensure required cols
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing required columns: {missing}")

    # Fill optional defaults
    for col, default in _OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    # Normalise condition labels
    df["condition"] = (
        df["condition"]
        .str.lower()
        .str.strip()
        .replace({"ctrl": "placebo", "control": "placebo", "sem": "semaglutide"})
    )

    # Derived quality score (0–1 scale assumed; clip just in case)
    for col in ("sharpness", "snr"):
        df[col] = df[col].clip(0, None)
        # min-max normalise per column if values outside [0,1]
        col_max = df[col].max()
        if col_max > 1.0:
            df[col] = df[col] / col_max

    df["quality_score"] = (df["sharpness"] + df["snr"]) / 2.0

    df = df.reset_index(drop=True)
    return df


# ── patch images ─────────────────────────────────────────────────────────────

_h5_handle: Optional[h5py.File] = None
_h5_index: Optional[dict[str, int]] = None


def _open_h5() -> tuple[h5py.File, dict[str, int]]:
    global _h5_handle, _h5_index
    if _h5_handle is None:
        _h5_handle = h5py.File(H5_PATH, "r")
        ids = _h5_handle["patch_ids"][:].astype(str).tolist()
        _h5_index = {pid: i for i, pid in enumerate(ids)}
    return _h5_handle, _h5_index


def load_patch_image(patch_id: str) -> Optional[Image.Image]:
    """
    Load a single 256×256 patch image.
    Tries:
      1. data/patches/<patch_id>.jpg  (or .png)
      2. data/patches.h5  dataset "images"[idx]
    Returns None if not found.
    """
    # Try flat file first
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        p = PATCHES_DIR / f"{patch_id}{ext}"
        if p.exists():
            return Image.open(p).convert("RGB")

    # Try HDF5
    if H5_PATH.exists():
        h5, idx_map = _open_h5()
        if patch_id in idx_map:
            arr = h5["images"][idx_map[patch_id]]  # (256,256,3) uint8
            return Image.fromarray(arr.astype(np.uint8))

    return None


def image_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()
