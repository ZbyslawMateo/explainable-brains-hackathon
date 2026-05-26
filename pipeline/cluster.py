"""
Multi-method clustering on patch embeddings.

Supported methods:
  hdbscan      — density-based, no k needed; min_cluster_size controls granularity
  kmeans       — k-means with explicit k; fast, equal-size clusters
  agglomerative — Ward hierarchical linkage with explicit k; compact clusters

Cache: one .npy file per method+params under cache/.
"""
from __future__ import annotations

import numpy as np
import hdbscan
from sklearn.cluster import KMeans, AgglomerativeClustering

from .cache import load_npy, save_npy

METHODS = ["HDBSCAN", "K-Means", "Agglomerative (Ward)"]


def compute_clusters(
    embeddings: np.ndarray,
    method: str = "HDBSCAN",
    *,
    # HDBSCAN
    min_cluster_size: int = 20,
    min_samples: int = 5,
    # K-Means / Agglomerative
    n_clusters: int = 25,
    random_state: int = 42,
    force: bool = False,
) -> np.ndarray:
    """
    Return integer array (N,) with cluster labels.
    -1 = noise (HDBSCAN only; K-Means and Agglomerative never produce -1).
    """
    if method == "HDBSCAN":
        cache_key = f"clusters_hdbscan_{min_cluster_size}_{min_samples}.npy"
    elif method == "K-Means":
        cache_key = f"clusters_kmeans_{n_clusters}.npy"
    elif method == "Agglomerative (Ward)":
        cache_key = f"clusters_agglomerative_{n_clusters}.npy"
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose from {METHODS}")

    if not force:
        cached = load_npy(cache_key)
        if cached is not None and cached.shape[0] == embeddings.shape[0]:
            return cached.astype(int)

    if method == "HDBSCAN":
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            core_dist_n_jobs=-1,
        )
        labels = clusterer.fit_predict(embeddings)

    elif method == "K-Means":
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        labels = km.fit_predict(embeddings)

    elif method == "Agglomerative (Ward)":
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = agg.fit_predict(embeddings)

    labels = labels.astype(np.int32)
    save_npy(cache_key, labels)
    return labels


def cluster_summary(labels: np.ndarray) -> dict[int, int]:
    """Return {cluster_id: count} for all clusters including noise (-1)."""
    unique, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


def cluster_centroids(embeddings: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """Return {cluster_id: centroid_vector} for all non-noise clusters."""
    centroids = {}
    for cl in set(labels.tolist()):
        if cl == -1:
            continue
        idxs = np.where(labels == cl)[0]
        centroids[int(cl)] = embeddings[idxs].mean(axis=0)
    return centroids
