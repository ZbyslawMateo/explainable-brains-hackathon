"""
Gaussian Mixture Model (GMM) clustering — soft probabilistic assignment.

Each patch gets a probability of belonging to each cluster rather than a hard label.
Better than KMeans when clusters have different sizes, shapes, or orientations.
The hard label is the argmax of the posterior probabilities.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GMMConfig:
    n_components: int = 8
    covariance_type: str = "full"   # 'full' | 'tied' | 'diag' | 'spherical'
    max_iter: int = 200
    n_init: int = 3                 # number of random inits — more = more stable
    random_state: int = 42


@dataclass
class ClusterResult:
    labels: np.ndarray              # hard assignment: argmax of posteriors
    proba: np.ndarray               # (N, k) soft probabilities
    n_clusters: int
    silhouette: float
    model: Any
    extra: dict = field(default_factory=dict)


def fit(features: np.ndarray, cfg: GMMConfig = GMMConfig()) -> ClusterResult:
    """
    Fit a GaussianMixture on a (N, D) feature matrix.

    Returns ClusterResult with:
      - labels     : (N,) int hard cluster assignments (argmax of posteriors)
      - proba      : (N, k) float soft membership probabilities
      - silhouette : silhouette score computed on hard labels
      - model      : fitted GaussianMixture object
      - extra      : {'bic': float, 'aic': float, 'converged': bool, 'n_iter': int}
    """
    gmm = GaussianMixture(
        n_components=cfg.n_components,
        covariance_type=cfg.covariance_type,
        max_iter=cfg.max_iter,
        n_init=cfg.n_init,
        random_state=cfg.random_state,
    )
    gmm.fit(features)
    labels = gmm.predict(features)
    proba  = gmm.predict_proba(features)

    sil = silhouette_score(features, labels, sample_size=min(2000, len(labels)), random_state=cfg.random_state)

    return ClusterResult(
        labels=labels,
        proba=proba,
        n_clusters=cfg.n_components,
        silhouette=sil,
        model=gmm,
        extra={
            "bic": gmm.bic(features),
            "aic": gmm.aic(features),
            "converged": gmm.converged_,
            "n_iter": gmm.n_iter_,
        },
    )


def sweep_k(features: np.ndarray, k_range=range(2, 17), cfg: GMMConfig = GMMConfig()) -> dict:
    """
    Sweep k values and return silhouette, BIC, and AIC scores.
    BIC/AIC penalise model complexity — lower is better.
    """
    results = {"k": [], "silhouette": [], "bic": [], "aic": []}
    for k in k_range:
        c = fit(features, GMMConfig(**{**cfg.__dict__, "n_components": k}))
        results["k"].append(k)
        results["silhouette"].append(c.silhouette)
        results["bic"].append(c.extra["bic"])
        results["aic"].append(c.extra["aic"])
    return results
