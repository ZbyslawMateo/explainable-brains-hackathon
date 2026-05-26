"""
Brain Patch Selector — professional research interface.
Run:  streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps

from pipeline.load import load_embeddings, load_metadata, load_patch_image
from pipeline.reduce import compute_projection, METHODS as PROJ_METHODS, METHOD_DESCRIPTIONS
from pipeline.cluster import compute_clusters, METHODS as CLUSTER_METHODS, cluster_summary
from pipeline.select import (
    select_subset, build_selection_info, coverage_score,
    find_similar_patches, STRATEGIES, SelectionInfo,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Brain Patch Selector",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Professional CSS ─────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Base */
html, body, [class*="css"] { font-family: "Inter", "Segoe UI", system-ui, sans-serif; }
.stApp { background: #0d0f18; }
section[data-testid="stSidebar"] { background: #11131e; border-right: 1px solid #1e2235; min-width: 300px !important; }

/* Remove Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; max-width: 100%; }

/* Section labels */
.section-label {
  font-size: 10px; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: #4b5280; margin: 0 0 6px 0;
  padding-top: 4px;
}

/* Compact metric */
.metric-row { display: flex; gap: 12px; margin-bottom: 10px; }
.metric-box {
  background: #161827; border: 1px solid #1e2235;
  border-radius: 6px; padding: 8px 12px; flex: 1; min-width: 0;
}
.metric-box .label { font-size: 10px; color: #4b5280; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-box .value { font-size: 20px; font-weight: 600; color: #c8cad8; line-height: 1.2; }
.metric-box .sub   { font-size: 11px; color: #6b7280; margin-top: 1px; }

/* Inspector card */
.inspector-card {
  background: #161827; border: 1px solid #1e2235;
  border-radius: 8px; padding: 14px 16px; margin-bottom: 10px;
}
.inspector-card .card-title {
  font-size: 10px; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; color: #4b5280; margin-bottom: 8px;
}

/* Selection badge */
.badge-selected   { background: #1a3a5c; color: #5b9cf5; border-radius: 4px; padding: 2px 7px; font-size: 11px; font-weight: 600; }
.badge-unselected { background: #1e2035; color: #6b7280; border-radius: 4px; padding: 2px 7px; font-size: 11px; font-weight: 500; }

/* Why-selected block */
.why-block {
  background: #0d1a2e; border-left: 3px solid #3b6bc4;
  border-radius: 0 6px 6px 0; padding: 10px 12px;
  font-size: 12px; color: #9ba3c4; line-height: 1.7;
  margin-top: 8px;
}
.why-block strong { color: #c8cad8; }

/* Metadata table */
.meta-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.meta-table td { padding: 4px 6px; color: #9ba3c4; }
.meta-table td:first-child { color: #6b7280; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; width: 45%; }
.meta-table td:last-child { color: #c8cad8; font-family: "JetBrains Mono", "Fira Mono", monospace; }
.meta-table tr:nth-child(odd) td { background: #0f1120; }

/* Stats bar */
.stats-bar {
  background: #11131e; border-top: 1px solid #1e2235;
  padding: 8px 20px; display: flex; gap: 24px; align-items: center;
  font-size: 12px; color: #6b7280; flex-wrap: wrap;
}
.stats-bar .stat-item strong { color: #c8cad8; }

/* Condition pills */
.pill-sema { background: #3a1c0e; color: #e8834a; border-radius: 3px; padding: 1px 6px; font-size: 11px; }
.pill-ctrl { background: #0e1c3a; color: #5b8af5; border-radius: 3px; padding: 1px 6px; font-size: 11px; }
.pill-noise { background: #1e2035; color: #6b7280; border-radius: 3px; padding: 1px 6px; font-size: 11px; }

/* Thin horizontal rule */
.thin-rule { border: none; border-top: 1px solid #1e2235; margin: 12px 0; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────

DATA_DIR    = Path("data")
CONDITION_COLOR = {"placebo": "#5b8af5", "semaglutide": "#e8834a", "unknown": "#6b7280"}

# Qualitative palette for clusters (cycles if >24 clusters)
CLUSTER_PALETTE = [
    "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
    "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac",
    "#499894","#86bcb6","#f1ce63","#f4e6c2","#d4a6c8",
    "#c7b4e2","#ffbe7d","#a0cbe8","#8cd17d","#b6992d",
    "#fabfd2","#d37295","#ff9888","#79706e","#d7b5a6",
]

# ── Session state init ───────────────────────────────────────────────────────

defaults = {
    "inspected_idx": None,       # row index of patch being inspected
    "img_vmin": 0,               # display window low  (0–127)
    "img_vmax": 255,             # display window high (128–255)
    "img_gamma": 1.0,            # gamma (0.3–3.0)
    "lasso_indices": [],         # manually lasso-selected indices
    "show_lasso": False,         # whether to show lasso result
    "gallery_page": 0,           # current gallery page index
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Data loading (cached) ─────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_embeddings():
    return load_embeddings()

@st.cache_data(show_spinner=False)
def _load_metadata():
    return load_metadata()

@st.cache_data(show_spinner="Computing projection…")
def _get_projection(method, n_neighbors, min_dist, perplexity, _emb_id):
    return compute_projection(
        _load_embeddings(), method=method,
        n_neighbors=n_neighbors, min_dist=min_dist, perplexity=perplexity,
    )

@st.cache_data(show_spinner="Clustering…")
def _get_clusters(method, min_cluster_size, min_samples, n_clusters, _emb_id):
    return compute_clusters(
        _load_embeddings(), method=method,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        n_clusters=n_clusters,
    )

@st.cache_data(show_spinner=False)
def _get_all_selections(target_n, cluster_method, cluster_params_hash, _emb_id):
    """Pre-compute all three selection strategies for instant switching."""
    emb  = _load_embeddings()
    meta = _load_metadata()
    labels = _get_clusters(*cluster_params_hash, _emb_id)
    return {
        s: select_subset(emb, meta, labels, s, target_n)
        for s in STRATEGIES
    }

# ── Check data ────────────────────────────────────────────────────────────────

if not (DATA_DIR / "embeddings.npy").exists() or not (DATA_DIR / "metadata.csv").exists():
    st.error("**Missing data.** Run `python download_data.py` first.")
    st.stop()

try:
    embeddings = _load_embeddings()
    metadata   = _load_metadata()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

N = len(embeddings)
emb_id = int(N)   # stable cache key (changes only if dataset changes)
patch_id_to_idx = {row["patch_id"]: i for i, row in enumerate(metadata.to_dict("records"))}

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 4px 0 12px 0;">
      <div style="font-size:15px; font-weight:700; color:#c8cad8; letter-spacing:-0.01em;">Brain Patch Selector</div>
      <div style="font-size:11px; color:#4b5280; margin-top:3px;">Active-learning data curation · c-Fos light-sheet</div>
    </div>
    """, unsafe_allow_html=True)

    n_brains = metadata["brain_id"].nunique()
    conditions = metadata["condition"].value_counts().to_dict()
    ctrl_n  = conditions.get("placebo", 0)
    sema_n  = conditions.get("semaglutide", 0)
    st.markdown(f"""
    <div style="font-size:12px; color:#6b7280; padding-bottom:12px; border-bottom:1px solid #1e2235;">
      <b style="color:#9ba3c4;">{N:,}</b> patches &nbsp;·&nbsp;
      <b style="color:#9ba3c4;">{n_brains}</b> brains &nbsp;·&nbsp;
      <span class="pill-ctrl">Vehicle {ctrl_n}</span>&nbsp;
      <span class="pill-sema">Semaglutide {sema_n}</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Projection ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Projection</p>', unsafe_allow_html=True)
    proj_method = st.radio(
        "Method", PROJ_METHODS, index=0,
        horizontal=True, label_visibility="collapsed",
    )
    st.caption(METHOD_DESCRIPTIONS[proj_method])

    proj_adv = st.expander("Projection parameters")
    with proj_adv:
        if proj_method == "UMAP":
            n_neighbors = st.slider("n_neighbors", 5, 50, 15,
                help="Higher = more global structure preserved")
            min_dist = st.slider("min_dist", 0.0, 0.9, 0.1, 0.05,
                help="Higher = looser clusters, less clumping")
            perplexity = 30.0
        elif proj_method == "t-SNE":
            perplexity = st.slider("Perplexity", 5, 100, 30,
                help="Roughly the expected cluster size")
            n_neighbors, min_dist = 15, 0.1
        else:
            n_neighbors, min_dist, perplexity = 15, 0.1, 30.0

    st.markdown("<hr class='thin-rule'>", unsafe_allow_html=True)

    # ── Clustering ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Clustering</p>', unsafe_allow_html=True)
    cluster_method = st.selectbox("Algorithm", CLUSTER_METHODS, label_visibility="collapsed")

    clust_adv = st.expander("Clustering parameters")
    with clust_adv:
        if cluster_method == "HDBSCAN":
            min_cluster_size = st.slider("Min cluster size", 5, 100, 20,
                help="Smaller = more clusters; larger = fewer, more stable clusters")
            min_samples = st.slider("Min samples", 1, 20, 5,
                help="Controls noise sensitivity. Higher = more points labelled noise")
            n_clusters = 25
        else:
            n_clusters = st.slider("Number of clusters (k)", 5, 80, 25)
            min_cluster_size, min_samples = 20, 5

    # Cache params as tuple for stable key
    cluster_params = (cluster_method, min_cluster_size, min_samples, n_clusters)

    st.markdown("<hr class='thin-rule'>", unsafe_allow_html=True)

    # ── Selection ───────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Selection</p>', unsafe_allow_html=True)
    strategy = st.selectbox("Strategy", STRATEGIES, index=2, label_visibility="collapsed")
    target_n = st.slider("Subset size", 50, min(1000, N), 300)

    text_query = st.text_input("Keyword filter",
        placeholder="e.g. bright, sharp, high-snr, dark",
        help="Narrows patches eligible for selection before the algorithm runs")

    st.markdown("<hr class='thin-rule'>", unsafe_allow_html=True)

    # ── Display ─────────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Display</p>', unsafe_allow_html=True)

    color_options = ["Selection status", "Cluster", "Condition", "Brain ID",
                     "Quality score", "SNR", "Sharpness", "Brightness"]
    color_by = st.selectbox("Colour by", color_options, label_visibility="collapsed")

    disp_col1, disp_col2 = st.columns(2)
    with disp_col1:
        point_size = st.slider("Point size", 2, 12, 4, label_visibility="visible")
    with disp_col2:
        unsel_opacity = st.slider("Unsel. opacity", 0.05, 0.8, 0.25, 0.05,
            label_visibility="visible")

    show_labels = st.toggle("Cluster centroids", value=False,
        help="Show cluster ID labels at centroid positions")
    show_density = st.toggle("Density contour", value=False,
        help="Overlay a 2D density estimate of all patches")

    st.markdown("<hr class='thin-rule'>", unsafe_allow_html=True)

    # ── Filters ─────────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Filters</p>', unsafe_allow_html=True)
    all_conditions = sorted(metadata["condition"].unique())
    all_brains     = sorted(metadata["brain_id"].unique())

    sel_conditions = st.multiselect("Condition", all_conditions, default=all_conditions,
        label_visibility="collapsed")
    with st.expander(f"Brain IDs ({len(all_brains)})"):
        sel_brains = st.multiselect("Brain", all_brains, default=all_brains,
            label_visibility="collapsed")

    st.markdown("<hr class='thin-rule'>", unsafe_allow_html=True)

    # ── Export ──────────────────────────────────────────────────────────────
    export_placeholder = st.empty()

# ═════════════════════════════════════════════════════════════════════════════
# COMPUTE PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

proj_coords  = _get_projection(proj_method, n_neighbors, min_dist, perplexity, emb_id)
cluster_labels = _get_clusters(*cluster_params, emb_id)

# Apply keyword filter
KEYWORD_RULES = {
    "bright":   ("brightness", "ge", 75),
    "dark":     ("brightness", "le", 25),
    "sharp":    ("sharpness",  "ge", 75),
    "blur":     ("sharpness",  "le", 25),
    "blurry":   ("sharpness",  "le", 25),
    "signal":   ("snr",        "ge", 75),
    "high-snr": ("snr",        "ge", 75),
    "noise":    ("snr",        "le", 25),
    "noisy":    ("snr",        "le", 25),
    "quality":  ("quality_score","ge",75),
    "contrast": ("contrast",   "ge", 75),
}
keyword_mask = pd.Series(True, index=metadata.index)
keyword_tags = []
if text_query:
    for word in text_query.lower().replace(",", " ").split():
        if word in KEYWORD_RULES:
            col, op, pct = KEYWORD_RULES[word]
            if col in metadata.columns:
                threshold = np.percentile(metadata[col], pct)
                if op == "ge":
                    keyword_mask &= metadata[col] >= threshold
                else:
                    keyword_mask &= metadata[col] <= threshold
                keyword_tags.append(f"{word}→{col}≥p{pct}" if op=="ge" else f"{word}→{col}≤p{pct}")

# All selections pre-computed
selections_all = _get_all_selections(target_n, cluster_method, cluster_params, emb_id)
selected_raw = selections_all[strategy]

# Apply keyword + condition + brain filter to selection
eligible_mask = (
    metadata["condition"].isin(sel_conditions) &
    metadata["brain_id"].isin(sel_brains) &
    keyword_mask
)
eligible_set = set(metadata[eligible_mask].index.tolist())
selected_indices = [i for i in selected_raw if i in eligible_set]
# Backfill if filter is too aggressive
if len(selected_indices) < max(10, target_n // 5):
    selected_indices = selected_raw
selected_set = set(selected_indices)

# Build SelectionInfo
quality_arr = metadata["quality_score"].values.astype(np.float32)
sel_info: dict[int, SelectionInfo] = build_selection_info(
    selected_indices, embeddings, cluster_labels, quality_arr, strategy
)

# ── Master DataFrame ──────────────────────────────────────────────────────────

df = metadata.copy()
df["umap_x"]   = proj_coords[:, 0]
df["umap_y"]   = proj_coords[:, 1]
df["cluster"]  = cluster_labels.astype(str).tolist()
df.loc[df["cluster"] == "-1", "cluster"] = "noise"
df["selected"] = df.index.isin(selected_set)
df["inspected"]= df.index == st.session_state.inspected_idx

# Merge lasso
if st.session_state.show_lasso and st.session_state.lasso_indices:
    lasso_set = set(st.session_state.lasso_indices)
    selected_set = selected_set | lasso_set
    df["selected"] = df.index.isin(selected_set)

n_selected = len(selected_set)
cov = coverage_score(list(selected_set), cluster_labels)

# ── Export button (now we know n_selected) ────────────────────────────────────

export_df = df[df["selected"]][
    ["patch_id","brain_id","condition","brightness","sharpness","contrast","snr","quality_score"]
].copy()
with export_placeholder:
    st.download_button(
        f"⬇  Export {n_selected} patches  (CSV)",
        data=export_df.to_csv(index=False).encode(),
        file_name="selected_patches.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# SCATTER FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def _color_series(df: pd.DataFrame, color_by: str):
    """Return (colors_array, colorscale_or_none, is_continuous)."""
    col_map = {
        "Selection status": "selected",
        "Condition":        "condition",
        "Brain ID":         "brain_id",
        "Cluster":          "cluster",
        "Quality score":    "quality_score",
        "SNR":              "snr",
        "Sharpness":        "sharpness",
        "Brightness":       "brightness",
    }
    col = col_map.get(color_by, "selected")

    if color_by == "Selection status":
        colors = df["selected"].map({True: "#5b9cf5", False: "#2a2d42"}).tolist()
        return colors, None, False

    if color_by == "Condition":
        colors = df["condition"].map(CONDITION_COLOR).fillna("#6b7280").tolist()
        return colors, None, False

    if color_by in ("Cluster", "Brain ID"):
        categories = sorted(df[col].unique())
        cat_colors = {c: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i, c in enumerate(categories)}
        colors = df[col].map(cat_colors).tolist()
        return colors, None, False

    # Continuous
    vals = df[col].values
    colors = vals.tolist()
    return colors, "Viridis", True


def make_scatter(df: pd.DataFrame, color_by: str, point_size: int,
                 unsel_opacity: float, show_labels: bool, show_density: bool) -> go.Figure:

    colors, colorscale, is_continuous = _color_series(df, color_by)
    df = df.copy()
    df["_color"] = colors

    # Build hover text before slicing into sel/unsel
    def hover_text(row):
        c = "#5b8af5" if row.condition == "placebo" else "#e8834a"
        return (
            f"<b>{row.patch_id}</b><br>"
            f"<span style='color:{c}'>{row.condition}</span> · {row.brain_id}<br>"
            f"SNR {row.snr:.3f} · Sharpness {row.sharpness:.3f}<br>"
            f"Quality {row.quality_score:.3f}"
        )

    df["_hover"] = [hover_text(r) for r in df.itertuples()]

    df_sel   = df[df["selected"]]
    df_unsel = df[~df["selected"]]

    fig = go.Figure()

    # Density contour
    if show_density:
        fig.add_trace(go.Histogram2dContour(
            x=df["umap_x"], y=df["umap_y"],
            colorscale=[[0,"rgba(0,0,0,0)"], [1,"rgba(100,120,200,0.15)"]],
            showscale=False, ncontours=8,
            contours=dict(coloring="fill"),
            line=dict(width=0),
        ))

    # Unselected points — slightly larger invisible border increases click target
    fig.add_trace(go.Scatter(
        x=df_unsel["umap_x"], y=df_unsel["umap_y"],
        mode="markers",
        marker=dict(
            color=df_unsel["_color"].tolist(),
            colorscale=colorscale if is_continuous else None,
            size=max(point_size, 5),      # minimum 5px so small dots are still clickable
            opacity=unsel_opacity,
            line=dict(width=0),
        ),
        customdata=list(zip(df_unsel.index, df_unsel["patch_id"])),
        text=df_unsel["_hover"],
        hoverinfo="text",
        name="Unselected",
        showlegend=False,
    ))

    # Selected points
    if not df_sel.empty:
        # Highlight the inspected patch with a white ring
        inspected_idx = st.session_state.inspected_idx
        df_sel_normal = df_sel[df_sel.index != inspected_idx]
        df_sel_focus  = df_sel[df_sel.index == inspected_idx]

        if not df_sel_normal.empty:
            fig.add_trace(go.Scatter(
                x=df_sel_normal["umap_x"], y=df_sel_normal["umap_y"],
                mode="markers",
                marker=dict(
                    color=df_sel_normal["_color"].tolist(),
                    colorscale=colorscale if is_continuous else None,
                    size=point_size + 4,
                    opacity=0.95,
                    symbol="circle",
                    line=dict(color="#ffffff", width=1.0),
                ),
                customdata=list(zip(df_sel_normal.index, df_sel_normal["patch_id"])),
                text=df_sel_normal["_hover"],
                hoverinfo="text",
                name="Selected",
                showlegend=False,
            ))

        if not df_sel_focus.empty:
            fig.add_trace(go.Scatter(
                x=df_sel_focus["umap_x"], y=df_sel_focus["umap_y"],
                mode="markers",
                marker=dict(
                    color="#ffffff",
                    size=point_size + 10,
                    symbol="circle",
                    line=dict(color="#5b9cf5", width=2.5),
                ),
                customdata=list(zip(df_sel_focus.index, df_sel_focus["patch_id"])),
                text=df_sel_focus["_hover"],
                hoverinfo="text",
                name="Inspected",
                showlegend=False,
            ))

    # Cluster centroid labels
    if show_labels:
        for cl in set(cluster_labels.tolist()):
            if cl == -1:
                continue
            idxs = np.where(cluster_labels == cl)[0]
            cx = float(df.iloc[idxs]["umap_x"].mean())
            cy = float(df.iloc[idxs]["umap_y"].mean())
            fig.add_annotation(
                x=cx, y=cy, text=str(cl),
                showarrow=False,
                font=dict(size=10, color="#6b7280"),
                bgcolor="rgba(0,0,0,0)",
            )

    axis_label = {"UMAP": ("UMAP 1", "UMAP 2"),
                  "t-SNE": ("t-SNE 1", "t-SNE 2"),
                  "PCA":   ("PC 1", "PC 2")}.get(proj_method, ("Dim 1", "Dim 2"))

    fig.update_layout(
        paper_bgcolor="#0d0f18",
        plot_bgcolor="#0d0f18",
        margin=dict(l=10, r=10, t=32, b=10),
        xaxis=dict(
            title=dict(text=axis_label[0], font=dict(size=11, color="#4b5280")),
            gridcolor="#161827", zerolinecolor="#161827",
            tickfont=dict(size=9, color="#4b5280"),
            showspikes=False,
        ),
        yaxis=dict(
            title=dict(text=axis_label[1], font=dict(size=11, color="#4b5280")),
            gridcolor="#161827", zerolinecolor="#161827",
            tickfont=dict(size=9, color="#4b5280"),
            showspikes=False,
        ),
        hoverlabel=dict(
            bgcolor="#161827", bordercolor="#2a2d42",
            font=dict(size=12, color="#c8cad8"),
        ),
        clickmode="event+select",   # single click on a point fires on_select
        dragmode="select",          # drag creates box/lasso selection
        uirevision="stable",        # preserve zoom/pan across reruns
        title=dict(
            text=f"<span style='font-size:12px;color:#6b7280'>"
                 f"{n_selected:,} of {N:,} patches selected "
                 f"· {cov:.0%} cluster coverage</span>",
            x=0.0, xanchor="left", yanchor="top",
            font=dict(size=12),
        ),
        height=580,
        newselection=dict(line=dict(color="#5b9cf5", width=1.5, dash="dot")),
        activeselection=dict(fillcolor="rgba(91,156,245,0.06)"),
        modebar=dict(
            bgcolor="rgba(0,0,0,0)", color="#4b5280", activecolor="#c8cad8",
            remove=["autoScale2d","resetScale2d","hoverCompareCartesian","hoverClosestCartesian"],
        ),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# INSPECTOR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def apply_display_adjustments(img: Image.Image, vmin: int, vmax: int, gamma: float) -> Image.Image:
    """Apply window/level and gamma to a grayscale image."""
    arr = np.array(img.convert("L"), dtype=np.float32)
    arr = np.clip(arr, vmin, vmax)
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr)
    if gamma != 1.0:
        arr = np.power(arr, gamma)
    return Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8), mode="L")


def img_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_why_selected(si: SelectionInfo) -> str:
    cl_str = f"Cluster {si.cluster_id}" if si.cluster_id != -1 else "Noise (unclustered)"
    lines = [
        f"<b>Strategy</b> &nbsp; {si.strategy}",
        f"<b>Step</b> &nbsp; {si.selection_step} of {len(selected_set)}",
        f"<b>{cl_str}</b> &nbsp; {si.cluster_size} patches",
    ]
    if si.cluster_id != -1:
        rank_text = "centroid representative" if si.cluster_rank == 1 else f"rank {si.cluster_rank}/{si.cluster_size} in cluster"
        lines.append(f"<b>Cluster rank</b> &nbsp; {rank_text}")
        lines.append(f"<b>Dist. to centroid</b> &nbsp; {si.distance_to_centroid:.3f}")
    lines.append(f"<b>Quality score</b> &nbsp; {si.quality_score:.3f} &nbsp; (top {100-si.quality_percentile:.0f}%)")
    if si.min_dist_to_neighbor > 0:
        lines.append(f"<b>Nearest selected</b> &nbsp; {si.min_dist_to_neighbor:.3f} away")
    return "<br>".join(lines)


def render_inspector(idx: int) -> None:
    row = metadata.iloc[idx]
    pid = row["patch_id"]
    is_sel = idx in selected_set
    si = sel_info.get(idx)

    # Header
    short_pid = pid.split("_MB1_")[0] if "_MB1_" in pid else pid[:30]
    cond_pill = (
        f'<span class="pill-sema">{row.condition}</span>'
        if row.condition == "semaglutide"
        else f'<span class="pill-ctrl">{row.condition}</span>'
    )
    badge = '<span class="badge-selected">✓ Selected</span>' if is_sel else '<span class="badge-unselected">Not selected</span>'
    st.markdown(f"""
    <div style="margin-bottom:10px;">
      <div style="font-size:12px; font-weight:600; color:#c8cad8; font-family:monospace;">{short_pid}</div>
      <div style="margin-top:4px; display:flex; gap:6px; align-items:center; flex-wrap:wrap;">
        {cond_pill}&nbsp;&nbsp;{badge}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Image + contrast controls
    st.markdown('<div class="inspector-card"><div class="card-title">Image · Window / Level</div>', unsafe_allow_html=True)
    img_raw = load_patch_image(pid)
    if img_raw:
        c1, c2 = st.columns([1, 1])
        with c1:
            vmin = st.slider("Min", 0, 200, st.session_state.img_vmin, key="img_vmin_slider",
                             label_visibility="visible")
            st.session_state.img_vmin = vmin
        with c2:
            vmax = st.slider("Max", 55, 255, st.session_state.img_vmax, key="img_vmax_slider",
                             label_visibility="visible")
            st.session_state.img_vmax = vmax
        gamma_val = st.slider("Gamma", 0.3, 3.0, st.session_state.img_gamma, 0.1, key="img_gamma_slider",
                              label_visibility="visible",
                              help="< 1.0 brightens dim signals; > 1.0 darkens")
        st.session_state.img_gamma = gamma_val

        display_img = apply_display_adjustments(img_raw, vmin, vmax, gamma_val)
        st.image(display_img, use_container_width=True, clamp=True)
    else:
        st.caption("Image file not found in data/patches/")
    st.markdown("</div>", unsafe_allow_html=True)

    # Why selected
    if is_sel and si:
        st.markdown(
            f'<div class="inspector-card"><div class="card-title">Why selected</div>'
            f'<div class="why-block">{render_why_selected(si)}</div></div>',
            unsafe_allow_html=True,
        )

    # Metadata table
    meta_rows = [
        ("SNR",             f"{row.snr:.4f}"),
        ("Sharpness",       f"{row.sharpness:.4f}"),
        ("Brightness",      f"{row.brightness:.4f}"),
        ("Contrast",        f"{row.contrast:.4f}"),
        ("Quality score",   f"{row.quality_score:.4f}"),
        ("Cluster",         df.iloc[idx]["cluster"]),
        ("Brain ID",        f"<span style='font-family:monospace;font-size:10px'>{row.brain_id}</span>"),
    ]
    rows_html = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in meta_rows
    )
    st.markdown(
        f'<div class="inspector-card"><div class="card-title">Metrics</div>'
        f'<table class="meta-table">{rows_html}</table></div>',
        unsafe_allow_html=True,
    )

    # Similar patches
    similar = find_similar_patches(idx, embeddings, n=6)
    if similar:
        st.markdown('<div class="inspector-card"><div class="card-title">Similar patches (embedding space)</div>', unsafe_allow_html=True)
        thumb_cols = st.columns(6)
        for col, (sim_idx, sim_score) in zip(thumb_cols, similar):
            sim_pid = metadata.iloc[sim_idx]["patch_id"]
            sim_img = load_patch_image(sim_pid)
            in_sel = "✓" if sim_idx in selected_set else ""
            if sim_img:
                display = apply_display_adjustments(sim_img, vmin, vmax, gamma_val)
                col.image(display.resize((80, 80)), use_container_width=True)
            col.caption(f"{in_sel} {sim_score:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATS (used across tabs)
# ══════════════════════════════════════════════════════════════════════════════

sel_df = df[df["selected"]]
n_cond = sel_df["condition"].value_counts().to_dict()
n_brains_sel = sel_df["brain_id"].nunique()
n_clusters_sel = len(set(cluster_labels[list(selected_set)].tolist()) - {-1})
n_clusters_total = len(set(cluster_labels.tolist()) - {-1})

rng = np.random.default_rng(42)
random_pick = rng.choice(N, size=min(n_selected, N), replace=False).tolist()
random_cov  = coverage_score(random_pick, cluster_labels)
cov_gain = cov - random_cov

sema_n_sel = n_cond.get("semaglutide", 0)
ctrl_n_sel  = n_cond.get("placebo", 0)

# ══════════════════════════════════════════════════════════════════════════════
# STATS BAR (always visible, above tabs)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="stats-bar">
  <span class="stat-item"><strong>{n_selected:,}</strong> / {N:,} patches selected</span>
  <span class="stat-item">Cluster coverage <strong>{cov:.0%}</strong>
    <span style="color:{'#4ade80' if cov_gain > 0 else '#6b7280'}; font-size:11px;">
      ({'+' if cov_gain >= 0 else ''}{cov_gain:.0%} vs random)
    </span>
  </span>
  <span class="stat-item">
    <span class="pill-ctrl">Vehicle {ctrl_n_sel}</span>&nbsp;
    <span class="pill-sema">Semaglutide {sema_n_sel}</span>
  </span>
  <span class="stat-item"><strong>{n_brains_sel}</strong> / {n_brains} brains</span>
  <span class="stat-item"><strong>{n_clusters_sel}</strong> / {n_clusters_total} clusters covered</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

tab_explorer, tab_gallery, tab_statistics = st.tabs(["🗺️ Explorer", "🖼️ Gallery", "📊 Statistics"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · EXPLORER — scatter plot + patch inspector
# ─────────────────────────────────────────────────────────────────────────────

with tab_explorer:
    col_scatter, col_inspector = st.columns([5, 3], gap="medium")

    with col_scatter:
        if keyword_tags:
            st.caption(f"Active keyword filters: {' · '.join(keyword_tags)}")

        fig = make_scatter(df, color_by, point_size, unsel_opacity, show_labels, show_density)
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            key="scatter",
            selection_mode=("points", "box", "lasso"),
        )

        # Handle click (single point) → inspector
        # Streamlit 1.35+ returns a typed SelectionState object; use attribute access
        try:
            sel_points = event.selection.points if event and hasattr(event, "selection") else []
        except Exception:
            sel_points = (event.get("selection", {}).get("points", [])
                          if isinstance(event, dict) else [])

        if sel_points:
            if len(sel_points) == 1:
                clicked_idx = int(sel_points[0]["customdata"][0])
                st.session_state.inspected_idx = clicked_idx
                st.session_state.show_lasso = False
            elif len(sel_points) > 1:
                lasso_idxs = [int(p["customdata"][0]) for p in sel_points]
                st.session_state.lasso_indices = lasso_idxs
                st.session_state.show_lasso = True

        if st.session_state.show_lasso and st.session_state.lasso_indices:
            n_lasso = len(st.session_state.lasso_indices)
            lc1, lc2, lc3 = st.columns([3, 1, 1])
            lc1.caption(f"Lasso: {n_lasso} patches captured")
            if lc2.button("Add to selection", key="lasso_add"):
                st.session_state.show_lasso = False
            if lc3.button("Clear", key="lasso_clear"):
                st.session_state.lasso_indices = []
                st.session_state.show_lasso = False

    with col_inspector:
        if st.session_state.inspected_idx is not None:
            render_inspector(st.session_state.inspected_idx)
        else:
            st.markdown("""
            <div style="height:100%; display:flex; flex-direction:column; justify-content:center;
                        align-items:center; padding:60px 20px; text-align:center;">
              <div style="font-size:28px; margin-bottom:12px; opacity:0.3">🔬</div>
              <div style="font-size:13px; color:#4b5280; line-height:1.7;">
                Click any point in the scatter<br>to inspect it here.<br><br>
                <span style="font-size:11px;">
                  Drag to lasso-select a region.<br>
                  Selected patches have a white outline.
                </span>
              </div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · GALLERY — thumbnail grid of selected patches
# ─────────────────────────────────────────────────────────────────────────────

with tab_gallery:
    if n_selected == 0:
        st.info("No patches selected. Adjust the subset size in the sidebar.")
    else:
        gallery_cols = st.columns([2, 1])
        with gallery_cols[1]:
            thumb_cols_n = st.selectbox(
                "Columns", [4, 6, 8, 10, 12], index=1,
                key="gallery_cols",
                label_visibility="visible",
            )
            sort_by = st.selectbox(
                "Sort by", ["Selection order", "Quality score ↓", "SNR ↓", "Sharpness ↓", "Cluster"],
                key="gallery_sort",
            )
        with gallery_cols[0]:
            st.markdown(
                f'<div style="font-size:12px; color:#6b7280; padding-top:8px;">'
                f'Showing <strong style="color:#c8cad8;">{n_selected}</strong> selected patches</div>',
                unsafe_allow_html=True,
            )

        # Sort selected dataframe
        gdf = sel_df.copy()
        gdf["_sel_order"] = gdf.index.map({idx: i for i, idx in enumerate(selected_indices)})
        if sort_by == "Quality score ↓":
            gdf = gdf.sort_values("quality_score", ascending=False)
        elif sort_by == "SNR ↓":
            gdf = gdf.sort_values("snr", ascending=False)
        elif sort_by == "Sharpness ↓":
            gdf = gdf.sort_values("sharpness", ascending=False)
        elif sort_by == "Cluster":
            gdf = gdf.sort_values("cluster")
        else:
            gdf = gdf.sort_values("_sel_order")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Paginate: 60 per page max
        page_size = thumb_cols_n * 10
        total_pages = max(1, (len(gdf) + page_size - 1) // page_size)
        if "gallery_page" not in st.session_state:
            st.session_state.gallery_page = 0
        st.session_state.gallery_page = min(st.session_state.gallery_page, total_pages - 1)

        page_start = st.session_state.gallery_page * page_size
        page_end   = min(page_start + page_size, len(gdf))
        page_df    = gdf.iloc[page_start:page_end]

        # Render thumbnail grid
        rows_iter = [page_df.iloc[i:i+thumb_cols_n] for i in range(0, len(page_df), thumb_cols_n)]
        vmin_g = st.session_state.img_vmin
        vmax_g = st.session_state.img_vmax
        gam_g  = st.session_state.img_gamma
        for row_slice in rows_iter:
            thumb_grid = st.columns(thumb_cols_n)
            for col_ui, (_, patch_row) in zip(thumb_grid, row_slice.iterrows()):
                pid = patch_row["patch_id"]
                img = load_patch_image(pid)
                if img:
                    display = apply_display_adjustments(img, vmin_g, vmax_g, gam_g)
                    thumb = display.resize((96, 96))
                    col_ui.image(thumb, use_container_width=True)
                else:
                    col_ui.markdown(
                        '<div style="width:100%;aspect-ratio:1;background:#1a1d2e;border-radius:4px;"></div>',
                        unsafe_allow_html=True,
                    )
                # Click to inspect: small button under each thumbnail
                short = pid.split("_")[-1] if "_" in pid else pid[:8]
                cond_icon = "🟣" if patch_row.get("condition") == "semaglutide" else "⚪"
                label = f"{cond_icon} #{short}"
                if col_ui.button(label, key=f"gal_{pid}", use_container_width=True):
                    # Find row index in master df
                    match = df[df["patch_id"] == pid]
                    if not match.empty:
                        st.session_state.inspected_idx = int(match.index[0])
                        st.rerun()

        # Pagination controls
        if total_pages > 1:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            pc1, pc2, pc3 = st.columns([1, 2, 1])
            with pc1:
                if st.button("← Prev", disabled=st.session_state.gallery_page == 0, key="gal_prev"):
                    st.session_state.gallery_page -= 1
                    st.rerun()
            with pc2:
                st.markdown(
                    f'<div style="text-align:center; font-size:12px; color:#6b7280; padding-top:6px;">'
                    f'Page {st.session_state.gallery_page + 1} / {total_pages}</div>',
                    unsafe_allow_html=True,
                )
            with pc3:
                if st.button("Next →", disabled=st.session_state.gallery_page >= total_pages - 1, key="gal_next"):
                    st.session_state.gallery_page += 1
                    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · STATISTICS — coverage, quality, condition breakdown
# ─────────────────────────────────────────────────────────────────────────────

with tab_statistics:
    sc1, sc2, sc3, sc4 = st.columns(4)
    def _kpi(col, label, value, sub=""):
        sub_html = (
            '<div style="font-size:10px; color:#4ade80; margin-top:2px;">' + sub + '</div>'
            if sub else ""
        )
        col.markdown(
            f'<div class="inspector-card" style="text-align:center; padding:16px 8px;">'
            f'<div style="font-size:22px; font-weight:700; color:#c8cad8;">{value}</div>'
            f'<div style="font-size:11px; color:#6b7280; margin-top:4px;">{label}</div>'
            f'{sub_html}'
            f'</div>',
            unsafe_allow_html=True,
        )
    _kpi(sc1, "Patches selected", f"{n_selected:,}", f"of {N:,} total")
    _kpi(sc2, "Cluster coverage", f"{cov:.0%}", f"{'+' if cov_gain >= 0 else ''}{cov_gain:.0%} vs random")
    _kpi(sc3, "Brains represented", f"{n_brains_sel} / {n_brains}")
    _kpi(sc4, "Clusters covered", f"{n_clusters_sel} / {n_clusters_total}")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    chart_left, chart_right = st.columns(2)

    # ── Condition breakdown bar ──────────────────────────────────────────────
    with chart_left:
        st.markdown('<p class="section-label">Condition breakdown — selected</p>', unsafe_allow_html=True)
        cond_labels = ["Vehicle (placebo)", "Semaglutide"]
        cond_values = [ctrl_n_sel, sema_n_sel]
        cond_colors = ["#4b88ff", "#a855f7"]
        fig_cond = go.Figure(go.Bar(
            x=cond_labels, y=cond_values,
            marker_color=cond_colors,
            text=cond_values, textposition="outside",
        ))
        fig_cond.update_layout(
            height=260, paper_bgcolor="#11131e", plot_bgcolor="#0d0f18",
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(color="#9ba3c4", size=11),
            yaxis=dict(gridcolor="#1e2235", showgrid=True),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_cond, use_container_width=True, key="stat_cond")

    # ── Patches per brain ────────────────────────────────────────────────────
    with chart_right:
        st.markdown('<p class="section-label">Patches per brain — selected</p>', unsafe_allow_html=True)
        brain_counts = sel_df["brain_id"].value_counts().sort_index()
        brain_conds  = sel_df.groupby("brain_id")["condition"].first()
        bar_colors   = ["#a855f7" if brain_conds.get(b) == "semaglutide" else "#4b88ff"
                        for b in brain_counts.index]
        fig_brain = go.Figure(go.Bar(
            x=brain_counts.index.tolist(),
            y=brain_counts.values.tolist(),
            marker_color=bar_colors,
            text=brain_counts.values.tolist(), textposition="outside",
        ))
        fig_brain.update_layout(
            height=260, paper_bgcolor="#11131e", plot_bgcolor="#0d0f18",
            margin=dict(l=10, r=10, t=10, b=30),
            font=dict(color="#9ba3c4", size=10),
            yaxis=dict(gridcolor="#1e2235", showgrid=True),
            xaxis=dict(showgrid=False, tickangle=-30),
        )
        st.plotly_chart(fig_brain, use_container_width=True, key="stat_brain")

    # ── Quality score distribution ───────────────────────────────────────────
    st.markdown('<p class="section-label">Quality score distribution</p>', unsafe_allow_html=True)
    qc1, qc2 = st.columns(2)
    with qc1:
        fig_q = go.Figure()
        fig_q.add_trace(go.Histogram(
            x=df["quality_score"].values, name="All patches",
            marker_color="#1e2235", opacity=0.8,
            nbinsx=60, histnorm="probability",
        ))
        fig_q.add_trace(go.Histogram(
            x=sel_df["quality_score"].values, name="Selected",
            marker_color="#4b88ff", opacity=0.7,
            nbinsx=60, histnorm="probability",
        ))
        fig_q.update_layout(
            barmode="overlay", height=240,
            paper_bgcolor="#11131e", plot_bgcolor="#0d0f18",
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(color="#9ba3c4", size=11),
            legend=dict(bgcolor="#11131e", bordercolor="#1e2235", borderwidth=1),
            yaxis=dict(gridcolor="#1e2235"),
            xaxis=dict(title="Quality score", showgrid=False),
        )
        st.plotly_chart(fig_q, use_container_width=True, key="stat_qual")

    with qc2:
        # SNR vs Sharpness scatter: all (grey) + selected (blue)
        fig_sq = go.Figure()
        sample_mask = np.random.default_rng(99).choice(N, size=min(2000, N), replace=False)
        sample_df = df.iloc[sample_mask]
        fig_sq.add_trace(go.Scatter(
            x=sample_df["snr"].values, y=sample_df["sharpness"].values,
            mode="markers", name="All (sample)",
            marker=dict(color="#1e2235", size=3, opacity=0.6),
        ))
        fig_sq.add_trace(go.Scatter(
            x=sel_df["snr"].values, y=sel_df["sharpness"].values,
            mode="markers", name="Selected",
            marker=dict(color="#4b88ff", size=5, opacity=0.8),
        ))
        fig_sq.update_layout(
            height=240,
            paper_bgcolor="#11131e", plot_bgcolor="#0d0f18",
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(color="#9ba3c4", size=11),
            legend=dict(bgcolor="#11131e", bordercolor="#1e2235", borderwidth=1),
            xaxis=dict(title="SNR", showgrid=False),
            yaxis=dict(title="Sharpness", gridcolor="#1e2235"),
        )
        st.plotly_chart(fig_sq, use_container_width=True, key="stat_snr_sharp")

    # ── Strategy comparison (coverage vs random) ─────────────────────────────
    st.markdown('<p class="section-label">Strategy comparison — cluster coverage vs random baseline</p>',
                unsafe_allow_html=True)
    strat_names, strat_covs = [], []
    for s in STRATEGIES:
        s_idxs = selections_all.get(s, [])
        if s_idxs:
            strat_names.append(s)
            strat_covs.append(coverage_score(s_idxs, cluster_labels))
    strat_names.append("Random")
    strat_covs.append(random_cov)

    bar_clrs = ["#4b88ff" if n != "Random" else "#374151" for n in strat_names]
    # Highlight current strategy
    bar_clrs = ["#a855f7" if n == strategy else c for n, c in zip(strat_names, bar_clrs)]
    fig_strat = go.Figure(go.Bar(
        x=strat_names, y=strat_covs,
        marker_color=bar_clrs,
        text=[f"{v:.0%}" for v in strat_covs], textposition="outside",
    ))
    fig_strat.update_layout(
        height=240,
        paper_bgcolor="#11131e", plot_bgcolor="#0d0f18",
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(color="#9ba3c4", size=11),
        yaxis=dict(gridcolor="#1e2235", tickformat=".0%", range=[0, 1.1]),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_strat, use_container_width=True, key="stat_strat")
