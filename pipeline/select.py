"""
Subset selection strategies + per-patch selection metadata.

select_subset()        → List[int]  (row indices)
build_selection_info() → Dict[int, SelectionInfo]  (why each patch was picked)
coverage_score()       → float
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ── public types ─────────────────────────────────────────────────────────────

STRATEGIES = ["Cluster-medoid", "Farthest-point", "Quality-weighted"]


@dataclass
class SelectionInfo:
    """Why a patch was included in the selection."""
    strategy: str
    selection_step: int        # 1-indexed order of selection
    cluster_id: int            # cluster label (-1 = noise)
    cluster_size: int          # patches in this cluster
    n_clusters_total: int      # non-noise clusters in dataset
    cluster_rank: int          # rank within cluster (1 = closest to centroid)
    distance_to_centroid: float
    min_dist_to_neighbor: float  # distance to nearest other selected patch
    quality_score: float
    quality_percentile: float   # 0–100 (100 = best)


# ── public API ────────────────────────────────────────────────────────────────

def select_subset(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    cluster_labels: np.ndarray,
    strategy: str,
    target_n: int,
    quality_col: str = "quality_score",
) -> list[int]:
    """
    Return row indices (up to target_n) for the selected subset.
    """
    target_n = min(target_n, len(embeddings))
    quality = metadata[quality_col].values.astype(np.float32)

    if strategy == "Cluster-medoid":
        return _cluster_medoid(embeddings, cluster_labels, target_n)
    elif strategy == "Farthest-point":
        return _farthest_point(embeddings, target_n)
    elif strategy == "Quality-weighted":
        return _quality_weighted(embeddings, quality, target_n)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Choose from {STRATEGIES}")


def build_selection_info(
    selected_indices: list[int],
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    quality: np.ndarray,
    strategy: str,
) -> dict[int, SelectionInfo]:
    """
    Compute per-patch explanatory metadata for every selected index.
    Returns {row_index: SelectionInfo}.
    """
    n_clusters_total = len(set(cluster_labels.tolist()) - {-1})
    centroids = _compute_centroids(embeddings, cluster_labels)

    # Build cluster_rank: within each cluster, rank selected patches by distance to centroid
    cluster_to_selected: dict[int, list[int]] = defaultdict(list)
    for idx in selected_indices:
        cluster_to_selected[int(cluster_labels[idx])].append(idx)

    cluster_ranks: dict[int, int] = {}
    for cl, idxs in cluster_to_selected.items():
        if cl == -1:
            for i, idx in enumerate(idxs, 1):
                cluster_ranks[idx] = i
        else:
            centroid = centroids[cl]
            dists = [(idx, float(np.linalg.norm(embeddings[idx] - centroid))) for idx in idxs]
            for rank, (idx, _) in enumerate(sorted(dists, key=lambda x: x[1]), 1):
                cluster_ranks[idx] = rank

    # Compute min-distance to nearest selected neighbour for each selected patch
    if len(selected_indices) > 1:
        sel_arr = np.array(selected_indices)
        sel_emb = embeddings[sel_arr]
        min_dists_to_neighbor: dict[int, float] = {}
        for i, idx in enumerate(selected_indices):
            d = np.linalg.norm(sel_emb - embeddings[idx], axis=1)
            d[i] = np.inf
            min_dists_to_neighbor[idx] = float(d.min())
    else:
        min_dists_to_neighbor = {selected_indices[0]: 0.0} if selected_indices else {}

    info: dict[int, SelectionInfo] = {}
    for step, idx in enumerate(selected_indices, 1):
        cl = int(cluster_labels[idx])
        cl_size = int((cluster_labels == cl).sum()) if cl != -1 else 1
        centroid = centroids.get(cl, embeddings[idx])
        q = float(quality[idx])
        q_pct = float(100.0 * (quality < q).mean())

        info[idx] = SelectionInfo(
            strategy=strategy,
            selection_step=step,
            cluster_id=cl,
            cluster_size=cl_size,
            n_clusters_total=n_clusters_total,
            cluster_rank=cluster_ranks.get(idx, 1),
            distance_to_centroid=float(np.linalg.norm(embeddings[idx] - centroid)),
            min_dist_to_neighbor=min_dists_to_neighbor.get(idx, 0.0),
            quality_score=q,
            quality_percentile=q_pct,
        )

    return info


def coverage_score(selected_indices: list[int], cluster_labels: np.ndarray) -> float:
    """Fraction of non-noise clusters represented in the selection."""
    unique = set(cluster_labels.tolist()) - {-1}
    if not unique:
        return 1.0
    covered = set(int(cluster_labels[i]) for i in selected_indices) - {-1}
    return len(covered) / len(unique)


def find_similar_patches(
    idx: int,
    embeddings: np.ndarray,
    n: int = 6,
) -> list[tuple[int, float]]:
    """
    Return top-n most similar patches (cosine similarity) to patch at idx.
    Embeddings are L2-normalised so cosine sim = dot product.
    Returns [(row_idx, similarity), ...], excluding idx itself.
    """
    sims = (embeddings @ embeddings[idx]).astype(float)
    sims[idx] = -np.inf
    top = np.argsort(sims)[::-1][:n]
    return [(int(i), float(sims[i])) for i in top]


# ── strategies ────────────────────────────────────────────────────────────────

def _cluster_medoid(embeddings: np.ndarray, labels: np.ndarray, target_n: int) -> list[int]:
    real_clusters = sorted(cl for cl in set(labels.tolist()) if cl != -1)
    noise_idxs = np.where(labels == -1)[0].tolist()
    selected: list[int] = []

    if real_clusters:
        cluster_sizes = {cl: int((labels == cl).sum()) for cl in real_clusters}
        total = sum(cluster_sizes.values())
        budget = min(target_n, total + min(len(noise_idxs), max(1, target_n // 10)))

        for cl in real_clusters:
            idxs = np.where(labels == cl)[0]
            alloc = max(1, round(budget * cluster_sizes[cl] / total))
            alloc = min(alloc, len(idxs))
            centroid = embeddings[idxs].mean(axis=0)
            dists = np.linalg.norm(embeddings[idxs] - centroid, axis=1)
            selected.extend(idxs[np.argsort(dists)[:alloc]].tolist())

    remaining = target_n - len(selected)
    if remaining > 0:
        selected.extend(noise_idxs[:remaining])

    return selected[:target_n]


def _farthest_point(embeddings: np.ndarray, target_n: int, seed: int = 0) -> list[int]:
    N = len(embeddings)
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(0, N))]
    min_dists = np.full(N, np.inf, dtype=np.float32)

    for _ in range(target_n - 1):
        last = embeddings[selected[-1]]
        dists = np.linalg.norm(embeddings - last, axis=1).astype(np.float32)
        np.minimum(min_dists, dists, out=min_dists)
        min_dists[selected] = 0.0
        selected.append(int(np.argmax(min_dists)))

    return selected


def _quality_weighted(
    embeddings: np.ndarray,
    quality: np.ndarray,
    target_n: int,
    quality_weight: float = 0.3,
    seed: int = 0,
) -> list[int]:
    N = len(embeddings)
    selected = [int(np.argmax(quality))]
    min_dists = np.full(N, np.inf, dtype=np.float32)

    for _ in range(target_n - 1):
        last = embeddings[selected[-1]]
        dists = np.linalg.norm(embeddings - last, axis=1).astype(np.float32)
        np.minimum(min_dists, dists, out=min_dists)
        min_dists[selected] = 0.0
        max_d = min_dists.max()
        norm_dists = min_dists / (max_d + 1e-8)
        score = (1.0 - quality_weight) * norm_dists + quality_weight * quality
        score[selected] = -1.0
        selected.append(int(np.argmax(score)))

    return selected


# ── internal helpers ─────────────────────────────────────────────────────────

def _compute_centroids(
    embeddings: np.ndarray, labels: np.ndarray
) -> dict[int, np.ndarray]:
    centroids: dict[int, np.ndarray] = {}
    for cl in set(labels.tolist()):
        if cl == -1:
            continue
        idxs = np.where(labels == cl)[0]
        centroids[int(cl)] = embeddings[idxs].mean(axis=0)
    return centroids
