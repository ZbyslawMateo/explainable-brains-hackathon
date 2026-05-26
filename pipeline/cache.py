"""
Disk cache helpers for expensive computations (UMAP, clustering).
"""
from __future__ import annotations
import os
import pickle
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def cache_path(name: str) -> Path:
    return CACHE_DIR / name


def save_npy(name: str, arr: np.ndarray) -> None:
    np.save(cache_path(name), arr)


def load_npy(name: str) -> np.ndarray | None:
    p = cache_path(name)
    if p.exists():
        return np.load(p)
    return None


def save_pkl(name: str, obj) -> None:
    with open(cache_path(name), "wb") as f:
        pickle.dump(obj, f)


def load_pkl(name: str):
    p = cache_path(name)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def invalidate(name: str) -> None:
    p = cache_path(name)
    if p.exists():
        p.unlink()
