# Clustering Backend — Handoff Notes for Frontend Integration

## What was built

A modular clustering pipeline in `src/` that takes precomputed image embeddings,
clusters them using one of three methods, and returns UMAP coordinates + labels
ready for visualisation. The notebook `clustering_comparison.ipynb` runs all three
methods and exports a single `interface_data` dict that the frontend can consume directly.

---

## Repo structure

```
explainable-brains-hackathon/
├── src/
│   ├── __init__.py
│   ├── cluster_kmeans.py          # KMeans clustering
│   ├── cluster_hierarchical.py    # Agglomerative / Hierarchical clustering
│   ├── cluster_gmm.py             # Gaussian Mixture Model clustering
│   └── run_clustering.py          # Unified entry point — import this
├── bucket_access/
│   ├── config.py                  # Hetzner S3 credentials (already set)
│   └── bucket_utils.py            # list_files, read_h5_embeddings, read_h5_patches
├── clustering_comparison.ipynb    # Main notebook — runs all 3 methods, exports interface_data
├── explore_challenge_a.ipynb      # Exploratory notebook (single method at a time)
└── claude/
    └── HANDOFF.md                 # This file
```

---

## Data

All data lives in a Hetzner S3 bucket. Credentials are in `bucket_access/config.py`.

```
bucket/challengeA/
├── embeddings/    12 × *_embeddings.h5   ~2 MB each   ← USE THESE for clustering
├── patches/       12 × *_patches.h5      ~50 MB each  ← only needed to display images
└── raw_whole_brain_data/                              ← do not download (~5 GB each)
```

**Embeddings** are 512-dim L2-normalised PLIP vectors (vision model trained on pathology
images). Patches that look similar produce similar vectors — cosine similarity = dot product.
Index `i` in `embeddings.h5` matches row `i` in `patches.h5` exactly.

Loading embeddings:
```python
from bucket_access.bucket_utils import list_files, read_h5_embeddings, read_h5_patches

emb_files = sorted([k for k in list_files('challengeA/embeddings/') if k.endswith('_embeddings.h5')])
embeddings, attrs = read_h5_embeddings(emb_files[0])
# embeddings.shape → (N, 512)  float32
# attrs → {'scan_name': ..., 'condition': 'Control' or 'Semaglutide', ...}
```

Loading metadata (patch quality stats — sharpness, SNR, etc.) without downloading images:
```python
_, metadata, _ = read_h5_patches(patch_files[0])
# metadata is a DataFrame with columns:
# patch_idx, z0, y0, x0, mean_intensity, std_intensity,
# fraction_signal, sharpness, snr, local_contrast, foreground_fraction
```

Loading raw patch images (only for display, ~50 MB):
```python
patches, metadata, attrs = read_h5_patches(patch_files[0])
# patches.shape → (N, 256, 256)  uint16
```

---

## The clustering API — `src/run_clustering.py`

### Quickstart

```python
from src.run_clustering import run, sweep_k, find_best_k, ClusteringConfig, Method

# Configure
cfg = ClusteringConfig(
    method       = Method.KMEANS,      # Method.KMEANS | Method.HIERARCHICAL | Method.GMM
    n_clusters   = 5,
    random_state = 42,
)

# Run — returns cluster labels + UMAP coordinates
result, umap1, umap2, dim_method = run(embeddings, metadata, cfg)

# result.labels      (N,)      int   — cluster assignment per patch
# result.silhouette  float           — quality score (-1 to 1, higher = better)
# result.model                       — fitted sklearn object
# umap1, umap2       (N,)      float — 2D projection coordinates
# dim_method         str             — 'UMAP' or 'PCA' (fallback if umap-learn missing)
```

### Auto-select best k (elbow method)

```python
sweep = sweep_k(embeddings, metadata, cfg, k_range=range(2, 17))
k_elbow, k_argmax = find_best_k(sweep, k_min=3)
# k_elbow   — elbow/kneedle recommendation (avoids trivial k=2)
# k_argmax  — highest raw silhouette score in range (often k=2, use as reference only)
```

### ClusteringConfig parameters

| Parameter | Default | Description |
|---|---|---|
| `method` | `Method.KMEANS` | Which algorithm to use |
| `n_clusters` | `8` | Number of clusters k |
| `metadata_weight` | `0.0` | Blend patch quality stats into features (0=embeddings only, 0.2–0.5=blend) |
| `metadata_features` | `['sharpness','snr','local_contrast','mean_intensity','foreground_fraction']` | Which stats to blend |
| `umap_n_neighbors` | `15` | UMAP local/global balance (5–50) |
| `umap_min_dist` | `0.1` | UMAP cluster tightness (0.0–0.3) |
| `kmeans_init` | `'k-means++'` | KMeans initialisation |
| `hierarchical_linkage` | `'ward'` | Linkage: `'ward'`\|`'complete'`\|`'average'`\|`'single'` |
| `gmm_covariance_type` | `'full'` | GMM covariance: `'full'`\|`'tied'`\|`'diag'`\|`'spherical'` |

### Method-specific extras

**GMM only** — `result.proba` is `(N, k)` float, soft membership probabilities per patch.
Use `result.proba.max(axis=1)` as a per-patch confidence score.

**KMeans** — `result.extra['cluster_centers']` is `(k, 512)` centroid matrix.

---

## The `interface_data` dict — what the frontend receives

Run all cells in `clustering_comparison.ipynb`. The last cell produces `interface_data`:

```python
interface_data = {
    'kmeans': {
        'labels':          np.ndarray (N,),        # int cluster per patch
        'n_clusters':      int,
        'silhouette':      float,
        'recommended_k':   int,                    # elbow auto-pick
        'representatives': np.ndarray (k,),        # patch index closest to each centroid
        'umap1':           np.ndarray (N,),        # x-axis for scatter
        'umap2':           np.ndarray (N,),        # y-axis for scatter
        'dim_method':      str,                    # 'UMAP' or 'PCA'
        'proba':           None,                   # None for KMeans/Hierarchical
    },
    'hierarchical': { ... },   # same shape
    'gmm': {
        ...
        'proba': np.ndarray (N, k),                # soft membership (GMM only)
    },
    '_shared': {
        'patches':   np.ndarray (N, 256, 256),     # uint16 raw images
        'metadata':  pd.DataFrame,                 # per-patch quality stats
        'scan_name': str,
        'condition': str,                          # 'Control' or 'Semaglutide'
        'n_clusters': int,
    }
}
```

---

## What the interface should do

The user flow intended for the frontend:

1. **Select a brain** — dropdown of the 12 scans (use `list_files('challengeA/embeddings/')`)
2. **Select clustering method** — radio/tabs: KMeans | Hierarchical | GMM
3. **Set k** — slider pre-filled with `recommended_k` from `interface_data`
4. **View UMAP scatter** — plot `umap1` vs `umap2`, colour points by `labels`
5. **Click a cluster** — show the representative patch image + nearby patches
6. **Compare methods** — side-by-side UMAP panel (already in the comparison notebook)

Optional extension from the challenge brief:
- Let the user draw on the UMAP or describe in natural language what signal they want to include/exclude — use that to filter which patches get forwarded to the AI training set.

---

## Recommended defaults (based on silhouette analysis)

Looking at the silhouette sweep across all 12 methods and this dataset:

- **Best overall method**: KMeans or GMM (similar scores, both outperform Hierarchical)
- **Recommended k**: 4–5 (elbow of the silhouette curve; k=8 loses ~15% silhouette score)
- **Hierarchical note**: Ward linkage struggles with 512-dim embeddings — if you use it, try `linkage='average'`
- **GMM advantage**: gives per-patch confidence scores (`proba`) useful for filtering uncertain patches

---

## Environment

```bash
# conda
conda env create -f environment.yml
conda activate explainable-brains

# or uv
uv venv --python 3.11 .venv
.venv\Scripts\Activate.ps1   # Windows
uv pip install -r requirements.txt
```

Key packages already in the environment: `streamlit`, `dash`, `plotly`, `umap-learn`,
`scikit-learn`, `h5py`, `s3fs`, `boto3`, `anthropic`.
