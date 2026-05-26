# Challenge A — Smart image data selection for generalizable AI models

## The problem

Modern AI models can be trained and tuned to solve problems across language, audio,
image, and video domains. Yet, the bottleneck often lies in selecting the right data,
in sufficient quality and quantity, to train models effectively. In practice, *less is
more* — a smaller, well-curated dataset often outperforms a large, noisy one. For
image-based signal detection, reliable ground truth is vital, and such label images are
typically generated through time-intensive semi-manual processes. The challenge,
therefore, is to automatically identify the most informative signal patterns that
represent the diversity of the dataset, enabling models to generalize well while
minimizing the need for manual labeling.

## Your goal

1. **Characterize** signal patterns in image patches extracted from whole-brain scans
2. **Select** an optimal subset of patches that captures the full diversity of signal patterns
3. **Build an interactive interface** that visualizes how selected patches relate to each
   other based on their characterization and clustering — making it easy to inspect and
   validate the final patch selection for AI model training

**Extension:** Allow users to guide patch selection by specifying what constitutes
relevant versus irrelevant signal, either through natural language descriptions or by
drawing directly on the images.

---

## The biology (brief)

The dataset comes from mouse brains imaged with **light sheet fluorescence microscopy**
at 5×5×5 µm voxel resolution. The marker is **c-Fos**, a protein expressed by recently
active neurons — a proxy for neuronal activation. Two conditions are compared:

- **G001 — Vehicle:** control mice
- **G002 — Semaglutide:** mice treated with semaglutide (active ingredient in Ozempic/Wegovy)

---

## Data

```
bucket/challengeA/
├── patches/               12 x *_patches.h5          one H5 file per brain
│                          all_patches_metadata.csv    metadata for all patches combined
├── embeddings/            12 x *_embeddings.h5        one H5 file per brain
└── raw_whole_brain_data/  4 x whole-brain H5 volumes  reference only — do not download
```

**Start with patches and embeddings — they are ready to use.**  
Raw whole-brain volumes are provided only as reference. Do not download them in full
(each is ~5 GB and not needed for this challenge).

---

## Patches — `challengeA/patches/`

One H5 file per brain (~500–700 patches each, 12 brains total, ~7500 patches overall).

Patches are 256×256 px 2D images (uint16) extracted as the middle slice of 256×256×64
voxel 3D subvolumes, sampled center-aligned across the whole brain. Only patches where
>70% of pixels exceed intensity 200 were kept — background and empty tissue are excluded.

### `all_patches_metadata.csv`

A combined CSV across all 12 brains for quick exploration without opening any H5 file.
Contains the same fields as the per-patch metadata inside each H5, plus `scan_name`,
`condition`, and `source_file` columns linking back to the originating H5.

```python
import pandas as pd
meta = pd.read_csv('all_patches_metadata.csv')
meta.groupby('condition')['patch_idx'].count()
meta.sort_values('sharpness', ascending=False).head(20)
```

### H5 file structure

```
{scan_name}_patches.h5
├── patches     (N, 256, 256)  uint16   — image patches
└── metadata    structured array        — one row per patch
```

### File-level attributes

| Attribute | Description |
|-----------|-------------|
| `scan_name` | Unique scan identifier |
| `animal_nr` | Animal identifier |
| `condition` | `Control` or `Semaglutide` |
| `voxel_size_um` | `[5, 5, 5]` µm |
| `patch_xy` | 256 px |
| `patch_z` | 64 slices |
| `signal_threshold` | 200 |
| `signal_fraction` | 0.70 |
| `n_patches` | Number of patches in file |

### Metadata fields (per patch)

| Field | Description |
|-------|-------------|
| `patch_idx` | Index into patches dataset — same index in embeddings file |
| `z0`, `y0`, `x0` | Origin coordinates in the original brain volume |
| `z_mid_absolute` | Absolute Z position of the saved 2D slice |
| `mean_intensity` | Mean pixel brightness |
| `std_intensity` | Spread of intensities |
| `fraction_signal` | Fraction of pixels above threshold (≥0.70 by definition) |
| `sharpness` | Laplacian variance — higher = sharper, lower = blurry |
| `snr` | Signal-to-noise ratio (mean / std) |
| `local_contrast` | Mean absolute gradient — higher = richer texture and edges |
| `foreground_fraction` | Tissue fraction via Otsu thresholding |

### Loading patches

```python
from bucket_access.bucket_utils import read_h5_patches
import pandas as pd

patches, metadata, attrs = read_h5_patches(
    "challengeA/patches/260219_AN0B7_G002_mouse_brain_MB1_SCAN0_16-11-05_patches.h5"
)

print(patches.shape)       # (N, 256, 256) uint16
print(metadata.head())
print(attrs['condition'])  # 'Semaglutide'
```

Load all 12 brains:

```python
from bucket_access.bucket_utils import list_files, read_h5_patches
import numpy as np
import pandas as pd

all_patches, all_meta = [], []

for key in sorted(list_files('challengeA/patches/')):
    if not key.endswith('_patches.h5'):
        continue
    patches, metadata, attrs = read_h5_patches(key)
    metadata['scan_name'] = attrs['scan_name']
    metadata['condition'] = attrs['condition']
    all_patches.append(patches)
    all_meta.append(metadata)

patches_all  = np.vstack(all_patches)
metadata_all = pd.concat(all_meta, ignore_index=True)
```

---

## Embeddings — `challengeA/embeddings/`

### What are embeddings and why are they here?

An embedding is a compact numerical representation of an image — a list of numbers that
captures its visual character. For each patch, PLIP (a vision model trained on pathology
images) produces a vector of 512 numbers. These numbers don't have individual meaning,
but **patches that look similar produce similar vectors** — so the distance between two
embeddings reflects how visually alike the patches are.

Embeddings are one way to characterize and compare patches without working with raw
pixels directly. They are provided as a convenience — **you are not required to use
them**. The challenge can be approached in many ways: using the precomputed embeddings,
using the patch metadata (intensity, sharpness, contrast, etc.), computing your own
image features, or any combination. Embeddings are simply there so you don't need a GPU
or waiting time to compute them yourself if you do want to use them.

Computing embeddings for ~7500 patches takes under a minute on a good GPU but over an
hour on a CPU laptop — which is why they are precomputed.

### Index alignment

Patch index `i` in `patches.h5` corresponds exactly to row `i` in `embeddings.h5`.
No joining needed — the index is the implicit key.

### File structure

```
{scan_name}_embeddings.h5
└── embeddings  (N, 512)  float32  — L2 normalized
```

| Attribute | Value |
|-----------|-------|
| `model` | `vinid/plip` |
| `embedding_dim` | 512 |
| `normalized` | `True` — L2 normalized, so cosine similarity = dot product |

### Loading embeddings

```python
from bucket_access.bucket_utils import read_h5_embeddings
import numpy as np

embeddings, attrs = read_h5_embeddings(
    "challengeA/embeddings/260219_AN0B7_G002_mouse_brain_MB1_SCAN0_16-11-05_embeddings.h5"
)

print(embeddings.shape)   # (N, 512) float32

# cosine similarity between patch 0 and patch 1 (L2 normalized = dot product)
sim = embeddings[0] @ embeddings[1]
```

Load all brains:

```python
from bucket_access.bucket_utils import list_files, read_h5_embeddings

all_emb = []
for key in sorted(list_files('challengeA/embeddings/')):
    if not key.endswith('_embeddings.h5'):
        continue
    emb, _ = read_h5_embeddings(key)
    all_emb.append(emb)

embeddings_all = np.vstack(all_emb)    # (N_total, 512) float32
```

---

## Raw whole-brain data — `challengeA/raw_whole_brain_data/` (reference only)

> ⚠️ Do not download full raw volumes. Each file is ~5 GB. Use patches and embeddings instead.
> Raw volumes are provided only if you want to understand the source data or extract
> patches yourself.

```python
from bucket_access.bucket_utils import read_h5_slice_remote, get_h5_info_remote

# check volume shape without downloading
info = get_h5_info_remote('challengeA/raw_whole_brain_data/260219_AN0B7_G002_mouse_brain_MB1_SCAN0_16-11-05.h5')
print(info['shape'])   # (Z, Y, X)

# read a single Z slice
single_slice = read_h5_slice_remote(
    'challengeA/raw_whole_brain_data/260219_AN0B7_G002_mouse_brain_MB1_SCAN0_16-11-05.h5',
    z_range=(500, 501)
)
```

| Field | Value |
|-------|-------|
| Format | HDF5 `.h5`, dataset key `data` |
| Shape | `(Z, Y, X)` ≈ `(1498, 2878, 2000)` |
| dtype | `uint16` |
| Voxel size | 5×5×5 µm |
| Conditions | 2 × Vehicle (G001), 2 × Semaglutide (G002) |
