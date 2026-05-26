# 🧠 Brain Patch Selector

Active-learning data curation for mouse brain scan images. Automatically selects the most **informative and diverse** patches for labelling — so you label less but train smarter.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2a. Use your real data (see Data format below)
#    OR generate synthetic test data first:
python generate_sample_data.py --n 500

# 3. Run the app
streamlit run app.py
```

## Data format

Place files in the `data/` directory:

| File | Description |
|---|---|
| `data/embeddings.npy` | Float32 array of shape `(N, 512)` — one embedding per patch |
| `data/metadata.csv` | CSV with columns: `patch_id`, `brain_id`, `condition` (placebo/semaglutide), `brightness`, `sharpness`, `contrast`, `snr` |
| `data/patches/` | 256×256 images named `<patch_id>.jpg` or `<patch_id>.png` |
| `data/patches.h5` _(alternative)_ | HDF5 with datasets `"images"` (N,256,256,3) and `"patch_ids"` (N,) |

## Features

### Selection strategies

| Strategy | How it works | Best for |
|---|---|---|
| **Cluster-medoid** | One representative from each visual cluster | Fast, interpretable |
| **FarthestPoint** | Greedy max-diversity in embedding space | Maximum geometric coverage |
| **Quality-weighted** | Diversity × quality score (sharpness × SNR) | Diverse _and_ usable images |

### Interactive explorer tabs

- **UMAP Explorer** — 2D scatter plot of all patches. Click any point to view the image + metadata.
- **Image Grid** — Thumbnail view of all selected patches.
- **Compare vs Random** — Coverage curve showing smart selection vs random baseline at each subset size.

### Text steering (bonus)

Type keywords in the sidebar to bias selection toward specific patch types:

```
bright        →  top-25% brightness
sharp         →  top-25% sharpness
signal        →  top-25% SNR
clean         →  top-25% SNR
noisy         →  bottom-25% SNR
dark          →  bottom-25% brightness
```

Multiple keywords can be combined: `bright, sharp, signal`

## Architecture

```
pipeline/
  load.py      — embeddings, metadata, lazy image access
  reduce.py    — UMAP 2D projection (cached to cache/)
  cluster.py   — HDBSCAN clustering (cached to cache/)
  select.py    — three selection strategies + coverage metric
app.py         — Streamlit UI
```

Results are cached to `cache/` — UMAP takes ~2 min on first run, instant thereafter.

## Key metric: Cluster coverage

> _What fraction of all discovered visual patterns is represented in your selection?_

A smart selection of 200 patches from 7,500 can cover **100% of clusters** — where a random draw of the same size might cover only 60%. This is the core value proposition shown in the Compare tab.

## Deploy

```bash
# Free deployment on Streamlit Community Cloud
# 1. Push to GitHub
# 2. Go to share.streamlit.io
# 3. Point to app.py
# Note: exclude data/ from git, upload separately or use st.file_uploader
```
