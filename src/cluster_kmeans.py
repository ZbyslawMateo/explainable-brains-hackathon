"""
KMeans clustering — centroid-based hard assignment.

Each patch is assigned to the nearest cluster centroid in feature space.
Good default: fast, scales well, works well with PLIP embeddings.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from dataclasses import dataclass, field
from typing import Any


@dataclass
class KMeansConfig:
    n_clusters: int = 8
    init: str = "k-means++"   # 'k-means++' or 'random'
    n_init: int | str = "auto"
    random_state: int = 42


@dataclass
class ClusterResult:
    labels: np.ndarray
    n_clusters: int
    silhouette: float
    model: Any
    extra: dict = field(default_factory=dict)


def fit(features: np.ndarray, cfg: KMeansConfig = KMeansConfig()) -> ClusterResult:
    """
    Fit KMeans on a (N, D) feature matrix.

    Returns ClusterResult with:
      - labels        : (N,) int cluster assignments
      - silhouette    : silhouette score (higher = better separated)
      - model         : fitted KMeans object
      - extra         : {'inertia': float, 'cluster_centers': (k, D)}
    """
    km = KMeans(
        n_clusters=cfg.n_clusters,
        init=cfg.init,
        n_init=cfg.n_init,
        random_state=cfg.random_state,
    )
    labels = km.fit_predict(features)
    sil = silhouette_score(features, labels, sample_size=min(2000, len(labels)), random_state=cfg.random_state)

    return ClusterResult(
        labels=labels,
        n_clusters=cfg.n_clusters,
        silhouette=sil,
        model=km,
        extra={"inertia": km.inertia_, "cluster_centers": km.cluster_centers_},
    )


def sweep_k(features: np.ndarray, k_range=range(2, 17), cfg: KMeansConfig = KMeansConfig()) -> dict:
    """
    Sweep k values and return silhouette scores and inertias.
    Useful for elbow / silhouette plots to pick a good k.
    """
    results = {"k": [], "silhouette": [], "inertia": []}
    for k in k_range:
        c = fit(features, KMeansConfig(**{**cfg.__dict__, "n_clusters": k}))
        results["k"].append(k)
        results["silhouette"].append(c.silhouette)
        results["inertia"].append(c.extra["inertia"])
    return results
