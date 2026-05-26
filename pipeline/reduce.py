"""
Dimensionality reduction: (N, D) → (N, 2).

Supported methods:
  umap   — UMAP (best for topology, ~2 min for 7 k patches, cached)
  tsne   — t-SNE with PCA pre-reduction to 50 dims (~3 min, cached)
  pca    — PCA (instant, linear, no cache needed but cached anyway)

Results cached under cache/ keyed by method + params.
"""
from __future__ import annotations

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .cache import load_npy, save_npy

METHODS = ["UMAP", "t-SNE", "PCA"]

# Human-readable descriptions shown in the sidebar
METHOD_DESCRIPTIONS = {
    "UMAP":  "Preserves local + global topology. Best for exploring cluster structure.",
    "t-SNE": "Emphasises local neighbourhood. Good for tight cluster separation.",
    "PCA":   "Linear projection. Fast, reproducible. Shows principal axes of variation.",
}


def compute_projection(
    embeddings: np.ndarray,
    method: str = "UMAP",
    *,
    # UMAP params
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    # t-SNE params
    perplexity: float = 30.0,
    # General
    random_state: int = 42,
    force: bool = False,
) -> np.ndarray:
    """
    Return (N, 2) float32 projection array.
    Loads from disk cache if available (unless force=True).
    """
    if method == "UMAP":
        cache_key = f"proj_umap_{n_neighbors}_{min_dist:.2f}.npy"
    elif method == "t-SNE":
        cache_key = f"proj_tsne_{perplexity:.0f}.npy"
    elif method == "PCA":
        cache_key = "proj_pca.npy"
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose from {METHODS}")

    if not force:
        cached = load_npy(cache_key)
        if cached is not None and cached.shape[0] == embeddings.shape[0]:
            return cached

    if method == "UMAP":
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric="cosine",
            random_state=random_state,
            low_memory=False,
        )
        coords = reducer.fit_transform(embeddings)

    elif method == "t-SNE":
        # PCA pre-reduction: t-SNE on raw 512-dim is very slow; 50 dims ≈ same quality
        n_pca = min(50, embeddings.shape[1])
        pca = PCA(n_components=n_pca, random_state=random_state)
        emb_reduced = pca.fit_transform(embeddings)
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            n_jobs=-1,
            init="pca",
            learning_rate="auto",
        )
        coords = tsne.fit_transform(emb_reduced)

    elif method == "PCA":
        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(embeddings)

    coords = coords.astype(np.float32)
    save_npy(cache_key, coords)
    return coords
