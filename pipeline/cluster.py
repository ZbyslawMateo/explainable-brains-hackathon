"""
Multi-method clustering on patch embeddings.

Supported methods:
  HDBSCAN             — density-based, no k needed; min_cluster_size controls granularity
  K-Means             — centroid-based hard assignment; fast, works well with PLIP embeddings
  GMM                 — soft probabilistic assignment; also returns per-patch confidence
  Agglomerative (Ward)— bottom-up hierarchical; compact clusters

Best-k selection:
  find_best_k(sweep)  — elbow/kneedle method on a silhouette sweep; avoids trivial k=2

Cache: one .npy file per method+params under cache/.
"""
from __future__ import annotations

import numpy as np
import hdbscan
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from dataclasses import dataclass

from .cache import load_npy, save_npy

METHODS = ["HDBSCAN", "K-Means", "GMM", "Agglomerative (Ward)"]


@dataclass
class ClusterResult:
    """Unified return type for all clustering methods."""
    labels: np.ndarray          # (N,) int  — cluster assignment (-1 = noise for HDBSCAN)
    silhouette: float | None    # None when not computed (e.g. HDBSCAN with noise points)
    proba: np.ndarray | None    # (N, k) float — soft membership (GMM only, else None)
    method: str


def compute_clusters(
    embeddings: np.ndarray,
    method: str = "HDBSCAN",
    *,
    # HDBSCAN
    min_cluster_size: int = 20,
    min_samples: int = 5,
    # K-Means / Agglomerative / GMM
    n_clusters: int = 25,
    # GMM
    covariance_type: str = "full",   # 'full' | 'tied' | 'diag' | 'spherical'
    gmm_n_init: int = 3,
    random_state: int = 42,
    force: bool = False,
) -> ClusterResult:
    """
    Cluster embeddings and return a ClusterResult.

    Labels:
      -1  = noise (HDBSCAN only)
      0…k = cluster index

    ClusterResult fields:
      .labels     (N,) int
      .silhouette float | None   — silhouette score; None if <2 valid clusters
      .proba      (N, k) | None  — GMM soft membership probabilities
      .method     str
    """
    # ── Cache key ──────────────────────────────────────────────────────────────
    if method == "HDBSCAN":
        cache_key = f"clusters_hdbscan_{min_cluster_size}_{min_samples}.npy"
    elif method == "K-Means":
        cache_key = f"clusters_kmeans_{n_clusters}.npy"
    elif method == "GMM":
        cache_key = f"clusters_gmm_{n_clusters}_{covariance_type}.npy"
    elif method == "Agglomerative (Ward)":
        cache_key = f"clusters_agglomerative_{n_clusters}.npy"
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose from {METHODS}")

    proba = None

    if not force:
        cached = load_npy(cache_key)
        if cached is not None and cached.shape[0] == embeddings.shape[0]:
            labels = cached.astype(int)
            sil = _silhouette(embeddings, labels)
            return ClusterResult(labels=labels, silhouette=sil, proba=None, method=method)

    # ── Fit ────────────────────────────────────────────────────────────────────
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

    elif method == "GMM":
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance_type,
            n_init=gmm_n_init,
            random_state=random_state,
        )
        gmm.fit(embeddings)
        labels = gmm.predict(embeddings)
        proba  = gmm.predict_proba(embeddings)   # (N, k) soft membership

    elif method == "Agglomerative (Ward)":
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = agg.fit_predict(embeddings)

    labels = labels.astype(np.int32)
    save_npy(cache_key, labels)

    sil = _silhouette(embeddings, labels)
    return ClusterResult(labels=labels, silhouette=sil, proba=proba, method=method)


def _silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float | None:
    """Compute silhouette score; returns None if not enough valid clusters."""
    valid = labels[labels >= 0]
    if len(np.unique(valid)) < 2:
        return None
    mask = labels >= 0
    try:
        return float(silhouette_score(
            embeddings[mask], labels[mask],
            sample_size=min(2000, mask.sum()),
            random_state=42,
        ))
    except Exception:
        return None


def find_best_k(
    embeddings: np.ndarray,
    method: str = "K-Means",
    k_range: range = range(2, 17),
    k_min: int = 3,
    random_state: int = 42,
) -> tuple[int, int]:
    """
    Sweep k values and return the elbow-method recommendation and argmax.

    Uses the kneedle algorithm (no extra dependency): normalises the silhouette
    curve to [0,1] and finds the point with maximum perpendicular distance from
    the line connecting the first and last points — that is the elbow.

    Parameters
    ----------
    embeddings : (N, D) feature matrix
    method     : 'K-Means', 'GMM', or 'Agglomerative (Ward)' (not HDBSCAN)
    k_range    : range of k values to sweep
    k_min      : ignore k values below this to avoid trivial k=2 answer

    Returns
    -------
    best_k_elbow  : int — elbow recommendation
    best_k_argmax : int — k with highest raw silhouette in [k_min, max_k]
    """
    ks, sils = [], []
    for k in k_range:
        result = compute_clusters(embeddings, method=method, n_clusters=k,
                                  random_state=random_state, force=True)
        if result.silhouette is not None:
            ks.append(k)
            sils.append(result.silhouette)

    ks_arr   = np.array(ks)
    sils_arr = np.array(sils)

    mask = ks_arr >= k_min
    ks_r, sils_r = ks_arr[mask], sils_arr[mask]

    # Normalise to [0,1]
    x = (ks_r - ks_r.min()) / (ks_r.max() - ks_r.min() + 1e-9)
    y = (sils_r - sils_r.min()) / (sils_r.max() - sils_r.min() + 1e-9)

    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    dx, dy = x1 - x0, y1 - y0
    distances = np.abs(dy * (x - x0) - dx * (y - y0)) / (np.sqrt(dx**2 + dy**2) + 1e-9)

    return int(ks_r[np.argmax(distances)]), int(ks_r[np.argmax(sils_r)])


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
