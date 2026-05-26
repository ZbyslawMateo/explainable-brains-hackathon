"""
run_clustering.py — unified entry point for all clustering methods.

Usage
-----
from src.run_clustering import run, ClusteringConfig, Method

result, umap1, umap2 = run(embeddings, metadata, cfg=ClusteringConfig(method=Method.GMM, n_clusters=10))

# result.labels    — (N,) int cluster assignments
# result.silhouette — float, quality score
# umap1, umap2     — (N,) 2D UMAP coordinates (or PCA fallback)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from sklearn.preprocessing import StandardScaler

from src import cluster_kmeans, cluster_hierarchical, cluster_gmm


class Method(str, Enum):
    KMEANS       = "kmeans"
    HIERARCHICAL = "hierarchical"
    GMM          = "gmm"


@dataclass
class ClusteringConfig:
    method: Method = Method.KMEANS

    # Shared
    n_clusters: int = 8
    random_state: int = 42

    # Feature blending
    metadata_features: list[str] = field(default_factory=lambda: [
        "sharpness", "snr", "local_contrast", "mean_intensity", "foreground_fraction"
    ])
    metadata_weight: float = 0.0   # 0 = embeddings only, 1 = metadata only

    # UMAP
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1

    # Method-specific overrides (optional)
    # KMeans
    kmeans_init: str = "k-means++"
    # Hierarchical
    hierarchical_linkage: str = "ward"
    # GMM
    gmm_covariance_type: str = "full"
    gmm_n_init: int = 3


def _build_features(
    embeddings: np.ndarray,
    metadata: pd.DataFrame | None,
    cfg: ClusteringConfig,
) -> np.ndarray:
    """Blend embeddings with optional metadata features into a single matrix."""
    if cfg.metadata_weight <= 0 or metadata is None:
        return embeddings.astype(np.float32)

    available = [f for f in cfg.metadata_features if f in metadata.columns]
    if not available:
        return embeddings.astype(np.float32)

    meta_arr  = StandardScaler().fit_transform(metadata[available].values.astype(np.float32))
    emb_w     = 1.0 - cfg.metadata_weight

    emb_norm  = embeddings / (np.linalg.norm(embeddings,  axis=1, keepdims=True) + 1e-8)
    meta_norm = meta_arr   / (np.linalg.norm(meta_arr,    axis=1, keepdims=True) + 1e-8)

    return np.hstack([emb_norm * emb_w, meta_norm * cfg.metadata_weight]).astype(np.float32)


def _compute_umap(embeddings: np.ndarray, cfg: ClusteringConfig) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (umap1, umap2, method_name). Falls back to PCA if umap-learn is absent."""
    try:
        import umap as umap_lib
        coords = umap_lib.UMAP(
            n_components=2,
            n_neighbors=cfg.umap_n_neighbors,
            min_dist=cfg.umap_min_dist,
            random_state=cfg.random_state,
            verbose=False,
        ).fit_transform(embeddings)
        return coords[:, 0], coords[:, 1], "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=cfg.random_state).fit_transform(embeddings)
        print("umap-learn not installed — using PCA. Install with: pip install umap-learn")
        return coords[:, 0], coords[:, 1], "PCA"


def run(
    embeddings: np.ndarray,
    metadata: pd.DataFrame | None = None,
    cfg: ClusteringConfig = ClusteringConfig(),
) -> tuple:
    """
    Run the selected clustering method and compute UMAP projections.

    Parameters
    ----------
    embeddings : (N, 512) float32  — L2-normalised PLIP embeddings
    metadata   : DataFrame with per-patch quality metrics (optional, used when metadata_weight > 0)
    cfg        : ClusteringConfig

    Returns
    -------
    result : ClusterResult from the chosen method module
    umap1  : (N,) float — first UMAP (or PCA) dimension
    umap2  : (N,) float — second UMAP (or PCA) dimension
    dim_method : str — 'UMAP' or 'PCA'
    """
    features = _build_features(embeddings, metadata, cfg)
    umap1, umap2, dim_method = _compute_umap(embeddings, cfg)

    method = Method(cfg.method)

    if method == Method.KMEANS:
        km_cfg = cluster_kmeans.KMeansConfig(
            n_clusters=cfg.n_clusters,
            init=cfg.kmeans_init,
            random_state=cfg.random_state,
        )
        result = cluster_kmeans.fit(features, km_cfg)

    elif method == Method.HIERARCHICAL:
        hc_cfg = cluster_hierarchical.HierarchicalConfig(
            n_clusters=cfg.n_clusters,
            linkage=cfg.hierarchical_linkage,
            random_state=cfg.random_state,
        )
        result = cluster_hierarchical.fit(features, hc_cfg)

    elif method == Method.GMM:
        gmm_cfg = cluster_gmm.GMMConfig(
            n_components=cfg.n_clusters,
            covariance_type=cfg.gmm_covariance_type,
            n_init=cfg.gmm_n_init,
            random_state=cfg.random_state,
        )
        result = cluster_gmm.fit(features, gmm_cfg)

    else:
        raise ValueError(f"Unknown method: {cfg.method}")

    print(
        f"[{method.value}]  k={cfg.n_clusters}  "
        f"silhouette={result.silhouette:.4f}  "
        f"dim_reduction={dim_method}"
    )
    return result, umap1, umap2, dim_method


def sweep_k(
    embeddings: np.ndarray,
    metadata: pd.DataFrame | None = None,
    cfg: ClusteringConfig = ClusteringConfig(),
    k_range=range(2, 17),
) -> dict:
    """
    Sweep k values for the selected method and return quality metrics.
    Useful for elbow / silhouette plots to pick the best k.
    """
    features = _build_features(embeddings, metadata, cfg)
    method   = Method(cfg.method)

    if method == Method.KMEANS:
        base_cfg = cluster_kmeans.KMeansConfig(init=cfg.kmeans_init, random_state=cfg.random_state)
        return cluster_kmeans.sweep_k(features, k_range, base_cfg)

    elif method == Method.HIERARCHICAL:
        base_cfg = cluster_hierarchical.HierarchicalConfig(linkage=cfg.hierarchical_linkage, random_state=cfg.random_state)
        return cluster_hierarchical.sweep_k(features, k_range, base_cfg)

    elif method == Method.GMM:
        base_cfg = cluster_gmm.GMMConfig(covariance_type=cfg.gmm_covariance_type, n_init=cfg.gmm_n_init, random_state=cfg.random_state)
        return cluster_gmm.sweep_k(features, k_range, base_cfg)

    raise ValueError(f"Unknown method: {cfg.method}")


def find_best_k(
    sweep: dict,
    k_min: int = 3,
) -> tuple[int, int]:
    """
    Pick the best k from a sweep_k result using two criteria.

    Strategy
    --------
    1. Elbow (kneedle-style) — finds where the silhouette curve bends most,
       i.e. the point with maximum perpendicular distance from the line
       connecting the first and last points. Avoids trivially picking k=2.
    2. Argmax silhouette in [k_min, max_k] — simple fallback / comparison.

    Parameters
    ----------
    sweep  : dict returned by sweep_k() — needs keys 'k' and 'silhouette'
    k_min  : ignore k values below this (default 3)

    Returns
    -------
    best_k_elbow  : int — elbow-method recommendation
    best_k_argmax : int — highest raw silhouette in [k_min, max_k]
    """
    ks   = np.array(sweep["k"])
    sils = np.array(sweep["silhouette"])

    mask = ks >= k_min
    ks_r, sils_r = ks[mask], sils[mask]

    # Normalise to [0,1] so axis scale doesn't bias geometry
    x = (ks_r - ks_r.min()) / (ks_r.max() - ks_r.min() + 1e-9)
    y = (sils_r - sils_r.min()) / (sils_r.max() - sils_r.min() + 1e-9)

    # Perpendicular distance from each point to the line [first → last]
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    dx, dy = x1 - x0, y1 - y0
    distances = np.abs(dy * (x - x0) - dx * (y - y0)) / (np.sqrt(dx**2 + dy**2) + 1e-9)

    best_k_elbow  = int(ks_r[np.argmax(distances)])
    best_k_argmax = int(ks_r[np.argmax(sils_r)])
    return best_k_elbow, best_k_argmax
