"""
Hierarchical (Agglomerative) clustering — bottom-up tree-based hard assignment.

Builds a dendrogram by merging the closest pairs of clusters step by step.
No need to re-run from scratch when changing k — the tree is computed once.
Useful for discovering nested structure and for comparing linkage strategies.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HierarchicalConfig:
    n_clusters: int = 8
    linkage: str = "ward"       # 'ward' | 'complete' | 'average' | 'single'
    metric: str = "euclidean"   # ignored when linkage='ward' (always euclidean)
    random_state: int = 42      # kept for API consistency; AgglomerativeClustering is deterministic


@dataclass
class ClusterResult:
    labels: np.ndarray
    n_clusters: int
    silhouette: float
    model: Any
    extra: dict = field(default_factory=dict)


def fit(features: np.ndarray, cfg: HierarchicalConfig = HierarchicalConfig()) -> ClusterResult:
    """
    Fit AgglomerativeClustering on a (N, D) feature matrix.

    Returns ClusterResult with:
      - labels     : (N,) int cluster assignments
      - silhouette : silhouette score
      - model      : fitted AgglomerativeClustering object
      - extra      : {'linkage': str, 'n_leaves': int, 'n_connected_components': int}
    """
    agg = AgglomerativeClustering(
        n_clusters=cfg.n_clusters,
        linkage=cfg.linkage,
        metric=cfg.metric if cfg.linkage != "ward" else "euclidean",
    )
    labels = agg.fit_predict(features)
    sil = silhouette_score(features, labels, sample_size=min(2000, len(labels)), random_state=cfg.random_state)

    return ClusterResult(
        labels=labels,
        n_clusters=cfg.n_clusters,
        silhouette=sil,
        model=agg,
        extra={
            "linkage": cfg.linkage,
            "n_leaves": agg.n_leaves_,
            "n_connected_components": agg.n_connected_components_,
        },
    )


def sweep_k(features: np.ndarray, k_range=range(2, 17), cfg: HierarchicalConfig = HierarchicalConfig()) -> dict:
    """Sweep k values and return silhouette scores."""
    results = {"k": [], "silhouette": []}
    for k in k_range:
        c = fit(features, HierarchicalConfig(**{**cfg.__dict__, "n_clusters": k}))
        results["k"].append(k)
        results["silhouette"].append(c.silhouette)
    return results
