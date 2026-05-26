"""
Download and prepare real challenge data from the Hetzner bucket.

Downloads:
  - all_patches_metadata.csv       (~small, combined across all 12 brains)
  - 12 × *_embeddings.h5           (~2 MB each  →  ~24 MB total)
  - 12 × *_patches.h5              (~50 MB each →  ~600 MB total)

Then prepares:
  data/embeddings.npy       (N_total, 512) float32
  data/metadata.csv         one row per patch with patch_id, brain_id, condition, ...
  data/patches/<id>.png     individual 256×256 uint8 PNGs (normalized for display)

Usage:
    python download_data.py                   # full download (~625 MB)
    python download_data.py --skip-patches    # embeddings + metadata only (~25 MB, no images)
    python download_data.py --brains 2        # first 2 brains only (quick test)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# ── Make sure bucket_access is importable ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from bucket_access.bucket_utils import (
    list_files,
    read_h5_patches,
    read_h5_embeddings,
    download_file,
)

DATA_DIR    = Path("data")
PATCHES_DIR = DATA_DIR / "patches"
DATA_DIR.mkdir(exist_ok=True)
PATCHES_DIR.mkdir(exist_ok=True)

# ── Metadata column mapping → names our app uses ────────────────────────────
# Real columns: mean_intensity, local_contrast, sharpness, snr
# App expects:  brightness,     contrast,       sharpness, snr
RENAME = {
    "mean_intensity": "brightness",
    "local_contrast": "contrast",
}

CONDITION_MAP = {
    "Control":     "placebo",
    "Vehicle":     "placebo",
    "Semaglutide": "semaglutide",
    "control":     "placebo",
    "vehicle":     "placebo",
    "semaglutide": "semaglutide",
}


def normalise_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Convert uint16 fluorescence image to display-ready uint8.
    Uses 1st–99th percentile clipping so bright spots don't wash out the image.
    """
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi <= lo:
        hi = lo + 1
    out = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def save_patch(arr_uint16: np.ndarray, patch_id: str) -> None:
    """Save a single uint16 patch as an 8-bit grayscale PNG."""
    img = normalise_to_uint8(arr_uint16)
    Image.fromarray(img, mode="L").save(PATCHES_DIR / f"{patch_id}.png")


def main(skip_patches: bool = False, max_brains: int | None = None) -> None:
    # ── Step 1: metadata CSV ─────────────────────────────────────────────────
    meta_csv = DATA_DIR / "metadata.csv"
    if not meta_csv.exists():
        print("Downloading all_patches_metadata.csv …")
        download_file(
            "challengeA/patches/all_patches_metadata.csv",
            str(meta_csv),
        )
    else:
        print(f"✓ metadata.csv already exists ({len(pd.read_csv(meta_csv)):,} rows)")

    # Load and normalise the metadata now
    meta_df = pd.read_csv(meta_csv)

    # Build patch_id = scan_name + "_" + zero-padded patch_idx
    meta_df["patch_id"] = (
        meta_df["scan_name"].astype(str)
        + "_"
        + meta_df["patch_idx"].astype(int).map(lambda x: f"{x:05d}")
    )
    meta_df["brain_id"] = meta_df["scan_name"]
    meta_df["condition"] = meta_df["condition"].map(CONDITION_MAP).fillna(meta_df["condition"])
    meta_df = meta_df.rename(columns=RENAME)

    # Keep only the columns the app needs (plus extras for exploration)
    keep = [
        "patch_id", "brain_id", "condition",
        "brightness", "sharpness", "contrast", "snr",
        "patch_idx", "scan_name", "z0", "y0", "x0",
        "fraction_signal", "foreground_fraction",
    ]
    keep = [c for c in keep if c in meta_df.columns]
    meta_df = meta_df[keep].reset_index(drop=True)

    print(f"Metadata: {len(meta_df):,} patches across {meta_df['brain_id'].nunique()} brains")
    print(meta_df["condition"].value_counts().to_string())
    meta_df.to_csv(meta_csv, index=False)
    print(f"✓ metadata.csv saved ({len(meta_df):,} rows)")

    # ── Step 2: embeddings ───────────────────────────────────────────────────
    emb_path = DATA_DIR / "embeddings.npy"
    if emb_path.exists():
        existing = np.load(emb_path)
        print(f"✓ embeddings.npy already exists {existing.shape}")
    else:
        emb_keys = sorted(k for k in list_files("challengeA/embeddings/") if k.endswith("_embeddings.h5"))
        if max_brains:
            emb_keys = emb_keys[:max_brains]
        print(f"\nDownloading {len(emb_keys)} embedding files …")

        all_emb = []
        downloaded_scans = []
        for key in emb_keys:
            emb, attrs = read_h5_embeddings(key)
            all_emb.append(emb)
            scan = str(attrs.get("scan_name", key.split("/")[-1].replace("_embeddings.h5", "")))
            downloaded_scans.append(scan)

        emb_all = np.vstack(all_emb).astype(np.float32)
        np.save(emb_path, emb_all)
        print(f"✓ embeddings.npy saved {emb_all.shape}")

        # Filter metadata to only the brains we actually downloaded
        if max_brains:
            meta_df = meta_df[meta_df["brain_id"].isin(downloaded_scans)].reset_index(drop=True)
            meta_df.to_csv(meta_csv, index=False)
            print(f"✓ metadata.csv trimmed to {len(meta_df):,} rows ({len(downloaded_scans)} brains)")

    # ── Step 3: patch images ─────────────────────────────────────────────────
    if skip_patches:
        print("\n--skip-patches set: skipping image extraction.")
        print("App will run without image click previews.")
        return

    patch_keys = sorted(k for k in list_files("challengeA/patches/") if k.endswith("_patches.h5"))
    if max_brains:
        patch_keys = patch_keys[:max_brains]

    print(f"\nDownloading + extracting {len(patch_keys)} patches files …")
    for key in patch_keys:
        patches, patch_meta, attrs = read_h5_patches(key)
        scan_name = str(attrs.get("scan_name", key.split("/")[-1].replace("_patches.h5", "")))
        n = patches.shape[0]
        print(f"  {scan_name}: {n} patches → writing PNGs …")

        for i in range(n):
            patch_id = f"{scan_name}_{i:05d}"
            out_path = PATCHES_DIR / f"{patch_id}.png"
            if not out_path.exists():
                save_patch(patches[i], patch_id)

        print(f"  ✓ {scan_name} done")

    total_pngs = len(list(PATCHES_DIR.glob("*.png")))
    print(f"\n✓ All done — {total_pngs:,} PNG files in data/patches/")
    print("Now delete cache/ and restart the app:  rm -rf cache && streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-patches", action="store_true",
                        help="Download only embeddings + metadata, skip image extraction (~25 MB)")
    parser.add_argument("--brains", type=int, default=None,
                        help="Limit to first N brains (for quick testing)")
    args = parser.parse_args()
    main(skip_patches=args.skip_patches, max_brains=args.brains)
