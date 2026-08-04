"""
Spatial Proteomics Threshold Tuner - Streamlit App
===================================================
Interactive tool to tune thresholds for marker-based cell type prediction
from multiplexed imaging data (zarr format).

Usage:
    streamlit run streamlit_app.py
"""

import os
import io
import base64
import warnings
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xarray as xr
import spatialproteomics as sp

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spatial Proteomics Threshold Tuner",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: black;
        color: white;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp li, .stApp span {
        color: white;
    }
    [data-testid="stSidebar"] {display: none !important;}
    section[data-testid="stSidebarContent"] {display: none !important;}
    .stMainBlockContainer, section.main > div.block-container {
        max-width: 100% !important;
        padding: 2rem 2rem 1rem 2rem !important;
    }
    [data-testid="stImage"] {
        width: 100% !important;
    }
    [data-testid="stImage"] img {
        width: 100% !important;
        height: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------
@st.cache_resource
def load_config(config_path="config.yml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


config = load_config()

ct_marker_dict = config["ct_marker_dict"]
all_celltype_colors = config["all_celltype_colors"]
all_marker_colors = config["all_marker_colors"]
marker_threshold_dict = config["marker_threshold_dict"]
threshold_csv_dir = config["threshold_csv_dir"]

# Markers used in the gating tree (level 1)
MARKERS = list(marker_threshold_dict.keys())
CELLTYPES = list(ct_marker_dict.values())

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def load_zarr_with_raw(zarr_path):
    """Load zarr dataset and attach the raw image layer (cached)."""
    ds = xr.open_zarr(zarr_path)
    ds = ds.pp.add_layer(
        ds["_image"].values.copy(), key_added="_raw_image"
    )
    return ds


@st.cache_data
def compute_initial_thresholds(_ds, marker_threshold_dict):
    """
    Compute initial threshold values based on quantile fractions.
    Returns a dict {marker: threshold_int}.
    """
    thresholds = {}
    for marker, fraction in marker_threshold_dict.items():
        img = _ds.pp[marker]["_image"].values
        threshold_int = int(np.percentile(img, 100 * fraction))
        thresholds[marker] = threshold_int
    return thresholds


@st.cache_data
def load_existing_thresholds(sample_id):
    """Load previously saved thresholds from CSV, if available."""
    csv_path = Path(threshold_csv_dir) / f"{sample_id}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0)
        return dict(zip(df["channel"], df["threshold"]))
    return None


# ---------------------------------------------------------------------------
# Core pipeline: threshold → quantify → transform → predict cell types
# ---------------------------------------------------------------------------
def run_pipeline(ds, thresholds_dict):
    """
    Run the full spatialproteomics pipeline with the given thresholds.
    Returns the processed dataset.
    """
    # 1. Threshold
    channels = list(thresholds_dict.keys())
    intensities = list(thresholds_dict.values())
    ds = ds.pp.threshold(
        intensity=intensities,
        channels=channels,
        key_added="_thresholded_image",  # use a different key to avoid merge conflicts with the original _image layer
    )

    # 2. Add quantification (intensity mean per cell) from the thresholded image
    ds = ds.pp.add_quantification(
        func="intensity_mean",
        layer_key="_thresholded_image",
    )

    # 3. Transform expression matrix (arcsinh)
    ds = ds.pp.transform_expression_matrix(method="arcsinh")

    # 4. Predict cell types (argmax)
    ds = ds.la.predict_cell_types_argmax(ct_marker_dict)

    # 5. Set label colors for plotting
    ds = ds.la.set_label_colors(
        list(all_celltype_colors.keys()),
        list(all_celltype_colors.values()),
    )

    # 6. Autocrop to remove empty space
    ds = ds.pl.autocrop()

    return ds


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------
# Major channels to show in the overview
OVERVIEW_CHANNELS = ["PAX5", "CD3", "CD11c", "CD11b", "CD68"]
# Major cell types to show in the predicted panel
OVERVIEW_CELLTYPES = ["BCell", "TCell", "Dendritic", "Myeloid", "Macro"]


# Downsample factor for display plots (reduce image resolution for speed)
def crop_ds(ds, y_start, y_end, x_start, x_end):
    """Crop dataset to the given region."""
    return ds.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))


def fig_to_rgb(fig, dpi=None):
    """Convert a matplotlib figure to an RGB numpy array (avoids PNG roundtrip)."""
    if dpi is not None:
        fig.set_dpi(dpi)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.renderer.buffer_rgba())
    return buf[:, :, :3]


def _fig_to_png_bytes(fig, dpi=None):
    """Render matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    save_kwargs = dict(format="png", bbox_inches="tight", pad_inches=0, facecolor=fig.get_facecolor())
    if dpi is not None:
        save_kwargs["dpi"] = dpi
    fig.savefig(buf, **save_kwargs)
    buf.seek(0)
    return buf.getvalue()


def plot_overview(ds):
    """
    Create the 4-panel overview figure as a static matplotlib figure.
    """
    ncols = 4
    scaling = 6
    fig, ax = plt.subplots(1, ncols, figsize=(ncols * scaling, scaling),
                           facecolor="black")
    ax = ax.flatten()
    for axis in ax:
        axis.set_facecolor("black")

    overview_colors = [all_marker_colors[ch] for ch in OVERVIEW_CHANNELS]

    # Panel 1: DAPI Segmentation
    ds.pp["DAPI"].pl.colorize("gold").pl.show(
        render_segmentation=True, ax=ax[0]
    )
    ax[0].set_title("DAPI Segmentation", fontsize=14, fontweight="bold", color="white")

    # Panel 2: Raw composite
    ds.pp[OVERVIEW_CHANNELS].pl.colorize(
        overview_colors, layer_key="_raw_image"
    ).pl.show(ax=ax[1])
    ax[1].set_title("RAW", fontsize=14, fontweight="bold", color="white")

    # Panel 3: Thresholded composite
    ds.pp[OVERVIEW_CHANNELS].pl.colorize(
        overview_colors, layer_key="_thresholded_image"
    ).pl.show(ax=ax[2])
    ax[2].set_title("Threshold", fontsize=14, fontweight="bold", color="white")

    # Panel 4: Predicted cell types
    try:
        ds.la[OVERVIEW_CELLTYPES].pl.show(
            render_image=False, render_labels=True, ax=ax[3]
        )
    except (ValueError, KeyError):
        ds.pl.show(render_image=False, render_labels=True, ax=ax[3])
    ax[3].set_title("Predicted", fontsize=14, fontweight="bold", color="white")

    for axis in ax:
        axis.axis("off")

    plt.tight_layout()
    return fig


def plot_marker_detail(ds, marker, color, celltype, threshold_value):
    """
    Create a 4-panel marker detail figure as a static matplotlib figure.
    """
    ncols = 4
    scaling = 5
    fig, ax = plt.subplots(1, ncols, figsize=(ncols * scaling, scaling),
                           facecolor="black")
    ax = ax.flatten()
    for axis in ax:
        axis.set_facecolor("black")

    # Panel 1: Marker raw + segmentation overlay
    try:
        ds.pp[marker].pl.colorize(color, layer_key="_raw_image").la[
            celltype
        ].pl.show(render_segmentation=True, ax=ax[0], legend_image=False, legend_segmentation=False, legend_label=False)
    except ValueError:
        ds.pp[marker].pl.colorize(color, layer_key="_raw_image").pl.show(
            render_segmentation=False, ax=ax[0], legend_image=False
        )
    ax[0].set_title(f"{marker} (raw+seg)", fontsize=12, fontweight="bold", color="white")

    # Panel 2: Marker raw
    ds.pp[marker].pl.colorize(color, layer_key="_raw_image").pl.show(ax=ax[1], legend_image=False)
    ax[1].set_title(f"{marker} (raw)", fontsize=12, fontweight="bold", color="white")

    # Panel 3: Marker thresholded
    ds.pp[marker].pl.colorize(color, layer_key="_thresholded_image").pl.show(ax=ax[2], legend_image=False)
    ax[2].set_title(f"{marker} (threshold: {threshold_value})", fontsize=12, fontweight="bold", color="white")

    # Panel 4: Cell type labels
    try:
        ds.la[celltype].pl.show(
            render_image=False, render_labels=True, ax=ax[3], legend_label=False
        )
    except ValueError:
        ax[3].imshow(np.zeros((100, 100)), cmap="gray", vmin=0, vmax=1)
    ax[3].set_title(f"{marker} (predict: {celltype})", fontsize=12, fontweight="bold", color="white")

    for axis in ax:
        axis.axis("off")

    plt.tight_layout()
    return fig


def _render_overview_raw(ds_processed, y_start, y_end, x_start, x_end):
    """Render overview plot to PNG bytes. Thread-safe (no st.session_state access)."""
    ds_cropped = crop_ds(ds_processed, y_start, y_end, x_start, x_end)
    fig = plot_overview(ds_cropped)
    dpi = 50 if st.session_state.get("low_data_mode") else 100
    png = _fig_to_png_bytes(fig, dpi=dpi)
    plt.close(fig)
    return png


def _render_detail_raw(ds_processed, marker, color, celltype, threshold_value, y_start, y_end, x_start, x_end):
    """Render per-marker detail plot to PNG bytes. Thread-safe (no st.session_state access)."""
    ds_cropped = crop_ds(ds_processed, y_start, y_end, x_start, x_end)
    fig = plot_marker_detail(ds_cropped, marker, color, celltype, threshold_value)
    dpi = 50 if st.session_state.get("low_data_mode") else 100
    png = _fig_to_png_bytes(fig, dpi=dpi)
    plt.close(fig)
    return png


@st.cache_data(max_entries=20, show_spinner=False)
def _render_overview_png(thresh_key, y_start, y_end, x_start, x_end):
    """Cached wrapper around _render_overview_raw."""
    return _render_overview_raw(st.session_state.ds_processed, y_start, y_end, x_start, x_end)


@st.cache_data(max_entries=100, show_spinner=False)
def _render_detail_png(thresh_key, marker, color, celltype, threshold_value, y_start, y_end, x_start, x_end):
    """Cached wrapper around _render_detail_raw."""
    return _render_detail_raw(st.session_state.ds_processed, marker, color, celltype,
                              threshold_value, y_start, y_end, x_start, x_end)


# ---------------------------------------------------------------------------
# Pipeline result cache (module-level) — shared between main script and fragments
# ---------------------------------------------------------------------------
def _make_thresh_key(thresholds_dict):
    return tuple(sorted((k, int(v)) for k, v in thresholds_dict.items()))


def _get_pipeline_cached(thresholds_dict):
    ds = st.session_state._base_ds
    key = _make_thresh_key(thresholds_dict)
    cache = st.session_state.pipeline_cache
    if key in cache:
        return cache[key]
    result = run_pipeline(ds, thresholds_dict)
    if len(cache) >= 15:
        oldest = next(iter(cache))
        del cache[oldest]
    cache[key] = result
    return result


def _img_to_b64_png(rgb):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@st.fragment
def render_marker_fragment(marker, color, celltype):
    thresholds = st.session_state.get("thresholds", {})
    threshold_value = thresholds.get(marker, 0)
    y_start, y_end = st.session_state.get("_zoom_y", (0, st.session_state._base_ds.dims["y"]))
    x_start, x_end = st.session_state.get("_zoom_x", (0, st.session_state._base_ds.dims["x"]))

    local_thresh = st.slider(
        f"Threshold — {marker}",
        min_value=0, max_value=255,
        value=int(threshold_value),
        key=f"marker_thresh_{marker}",
    )

    thresholds[marker] = local_thresh
    st.session_state.thresholds = thresholds

    local_thresholds = thresholds.copy()
    local_thresholds[marker] = local_thresh
    thresh_key = _make_thresh_key(local_thresholds)

    # --- Detail image ---
    if local_thresh == threshold_value:
        detail_png = st.session_state.rendered_detail_pngs.get(marker) if st.session_state.get("rendered_detail_pngs") else None
        if detail_png is None:
            ds_processed = st.session_state.get("ds_processed")
            if ds_processed is not None:
                detail_png = _render_detail_raw(ds_processed, marker, color, celltype, threshold_value,
                                                y_start, y_end, x_start, x_end)
        if detail_png is not None:
            st.image(detail_png)
    else:
        if "zoom_detail_cache" not in st.session_state:
            st.session_state.zoom_detail_cache = {}
        zcache = st.session_state.zoom_detail_cache
        cache_key = (thresh_key, marker)
        if cache_key in zcache:
            detail_png = zcache[cache_key]
        else:
            ds_local = _get_pipeline_cached(local_thresholds)
            detail_png = _render_detail_raw(ds_local, marker, color, celltype, local_thresh,
                                            y_start, y_end, x_start, x_end)
            if len(zcache) >= 80:
                old = list(zcache.keys())[:20]
                for ok in old:
                    del zcache[ok]
            zcache[cache_key] = detail_png
        st.image(detail_png)

    # --- Interactive zoom expander ---
    with st.expander(f"🔍 Zoom: {marker} → {celltype}", expanded=False):
        st.markdown(
            f"**{marker}** — drag/scroll to zoom, double-click to reset."
        )

        with st.spinner(f"Re-running pipeline for {marker}..."):
            try:
                cache_keys = [
                    (thresh_key, marker, "raw"),
                    (thresh_key, marker, "thresh"),
                    (thresh_key, marker, "pred"),
                ]
                cached_images = st.session_state.zoom_img_cache

                if all(k in cached_images for k in cache_keys):
                    b64_raw, b64_thresh, b64_pred = (cached_images[k] for k in cache_keys)
                else:
                    ds_local = _get_pipeline_cached(local_thresholds)
                    ds_local_cropped = crop_ds(ds_local, y_start, y_end, x_start, x_end)

                    def _render_panel(ds_data, channel, panel_color, layer_key,
                                      render_segmentation=False, render_labels=False, ct=None):
                        fig, ax = plt.subplots(1, 1, figsize=(3, 3), facecolor="black")
                        ax.set_facecolor("black")
                        ax.axis("off")
                        try:
                            if render_labels and ct:
                                ds_data.la[ct].pl.show(render_image=False, render_labels=True, ax=ax, legend_label=False)
                            elif render_segmentation:
                                ds_data.pp[channel].pl.colorize(panel_color, layer_key=layer_key).la[ct].pl.show(
                                    render_segmentation=True, ax=ax, legend_image=False, legend_segmentation=False, legend_label=False
                                )
                            else:
                                ds_data.pp[channel].pl.colorize(panel_color, layer_key=layer_key).pl.show(ax=ax, legend_image=False)
                        except (ValueError, KeyError, AttributeError):
                            ax.imshow(np.zeros((100, 100)), cmap="gray", vmin=0, vmax=1)
                        plt.tight_layout(pad=0)
                        zoom_dpi = 40 if st.session_state.get("low_data_mode") else 72
                        rgb = fig_to_rgb(fig, dpi=zoom_dpi)
                        plt.close(fig)
                        return rgb

                    b64_raw = _img_to_b64_png(_render_panel(ds_local_cropped, marker, color, "_raw_image"))
                    b64_thresh = _img_to_b64_png(_render_panel(ds_local_cropped, marker, color, "_thresholded_image"))
                    b64_pred = _img_to_b64_png(_render_panel(ds_local_cropped, marker, color, "_thresholded_image",
                                               render_labels=True, ct=celltype))

                    if len(cached_images) >= st.session_state.zoom_img_cache_max:
                        old_keys = list(cached_images.keys())[:40]
                        for ok in old_keys:
                            del cached_images[ok]
                    for k, v in zip(cache_keys, [b64_raw, b64_thresh, b64_pred]):
                        cached_images[k] = v

                html = f"""
                <style>
                .zoom-grid-{marker} {{
                    display: grid;
                    grid-template-columns: 1fr 1fr 1fr;
                    gap: 8px;
                    max-width: 100%;
                }}
                .zoom-cell-{marker} {{
                    position: relative;
                    overflow: hidden;
                    background: black;
                    border: 1px solid #555;
                }}
                .zoom-cell-{marker} img {{
                    width: 100%;
                    height: auto;
                    display: block;
                    cursor: crosshair;
                    transition: none;
                }}
                </style>
                <div class="zoom-grid-{marker}" id="sync-zoom-{marker}">
                    <div class="zoom-cell-{marker}">
                        <img id="img-{marker}-0" src="data:image/png;base64,{b64_raw}" data-scale="1" data-ox="0.5" data-oy="0.5">
                    </div>
                    <div class="zoom-cell-{marker}">
                        <img id="img-{marker}-1" src="data:image/png;base64,{b64_thresh}" data-scale="1" data-ox="0.5" data-oy="0.5">
                    </div>
                    <div class="zoom-cell-{marker}">
                        <img id="img-{marker}-2" src="data:image/png;base64,{b64_pred}" data-scale="1" data-ox="0.5" data-oy="0.5">
                    </div>
                </div>
                <script>
                (function() {{
                    const prefix = 'img-{marker}-';
                    const images = [];
                    for (let i = 0; i < 3; i++) {{
                        const img = document.getElementById(prefix + i);
                        if (img) images.push(img);
                    }}
                    if (images.length === 0) return;
                    let isDragging = false;
                    let startX, startY;
                    let currentScale = 1;
                    let originX = 0.5, originY = 0.5;
                    function applyTransform(scale, ox, oy) {{
                        images.forEach(img => {{
                            img.style.transformOrigin = (ox * 100) + '% ' + (oy * 100) + '%';
                            img.style.transform = 'scale(' + scale + ')';
                            img.dataset.scale = scale;
                            img.dataset.ox = ox;
                            img.dataset.oy = oy;
                        }});
                    }}
                    function handleWheel(e) {{
                        e.preventDefault();
                        const rect = e.target.getBoundingClientRect();
                        const ox = (e.clientX - rect.left) / rect.width;
                        const oy = (e.clientY - rect.top) / rect.height;
                        const delta = e.deltaY > 0 ? -0.1 : 0.1;
                        currentScale = Math.max(1, Math.min(4, currentScale + delta));
                        originX = ox;
                        originY = oy;
                        applyTransform(currentScale, originX, originY);
                    }}
                    function handleMouseDown(e) {{
                        if (currentScale > 1) {{
                            isDragging = true;
                            startX = e.clientX;
                            startY = e.clientY;
                            e.target.style.cursor = 'grabbing';
                        }}
                    }}
                    function handleMouseMove(e) {{
                        if (isDragging && currentScale > 1) {{
                            const dx = (e.clientX - startX) / e.target.width;
                            const dy = (e.clientY - startY) / e.target.height;
                            originX = Math.max(0, Math.min(1, originX - dx));
                            originY = Math.max(0, Math.min(1, originY - dy));
                            startX = e.clientX;
                            startY = e.clientY;
                            applyTransform(currentScale, originX, originY);
                        }}
                    }}
                    function handleMouseUp(e) {{
                        if (isDragging) {{
                            isDragging = false;
                            e.target.style.cursor = 'crosshair';
                        }} else if (currentScale === 1) {{
                            const rect = e.target.getBoundingClientRect();
                            originX = (e.clientX - rect.left) / rect.width;
                            originY = (e.clientY - rect.top) / rect.height;
                            currentScale = 2;
                            applyTransform(currentScale, originX, originY);
                        }}
                    }}
                    function handleDblClick(e) {{
                        currentScale = 1;
                        originX = 0.5;
                        originY = 0.5;
                        applyTransform(1, 0.5, 0.5);
                    }}
                    images.forEach(img => {{
                        img.addEventListener('wheel', handleWheel, {{passive: false}});
                        img.addEventListener('mousedown', handleMouseDown);
                        document.addEventListener('mousemove', handleMouseMove);
                        document.addEventListener('mouseup', handleMouseUp);
                        img.addEventListener('dblclick', handleDblClick);
                    }});
                }})();
                </script>
                """
                st.components.v1.html(html, height=600, scrolling=True)

            except Exception as e:
                st.error(f"Error in zoom view for {marker}: {e}")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.title("🧬 Spatial Proteomics Threshold Tuner")
    st.markdown(
        "Interactive tool for tuning marker thresholds used in cell type prediction. "
        "Adjust the slider above each marker and click **Run Pipeline** to see results."
    )

    # ------------------------------------------------------------------
    # Session state defaults
    # ------------------------------------------------------------------
    if "path_counter" not in st.session_state:
        st.session_state.path_counter = 0
    if "pipeline_executed" not in st.session_state:
        st.session_state.pipeline_executed = False
    if "low_data_mode" not in st.session_state:
        st.session_state.low_data_mode = False

    # ------------------------------------------------------------------
    # TOP CONTROLS
    # ------------------------------------------------------------------
    zarr_path = st.text_input(
        "📁 Zarr file path",
        key=f"zarr_path_input_{st.session_state.path_counter}",
        placeholder="e.g. /path/to/data.zarr",
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    run_label = "🪂 Re-run Pipeline" if st.session_state.pipeline_executed else "🪂 Run Pipeline"
    run_top = col1.button(run_label, type="primary", use_container_width=True)

    if st.session_state.pipeline_executed:
        if col2.button("🔄 Reload", use_container_width=True):
            st.session_state.path_counter += 1
            for k in ("pipeline_has_run", "ds_processed", "pipeline_executed",
                       "rendered_overview_png", "rendered_detail_pngs",
                       "_rendered_thresh_key", "zoom_img_cache", "zoom_detail_cache"):
                st.session_state.pop(k, None)
            st.rerun()
    else:
        clear_disabled = not bool(zarr_path)
        if col2.button("🗑️ Clear Path", disabled=clear_disabled, use_container_width=True):
            st.session_state.path_counter += 1
            st.rerun()

    low_data = col3.checkbox(
        "Low Data Mode",
        disabled=not st.session_state.pipeline_executed,
        value=st.session_state.low_data_mode,
    )
    if low_data != st.session_state.low_data_mode:
        st.session_state.low_data_mode = low_data
        st.rerun()

    st.divider()

    # ------------------------------------------------------------------
    # Validate path
    # ------------------------------------------------------------------
    if not zarr_path:
        st.info("👆 Enter a zarr file path above to get started.")
        return
    if not os.path.exists(zarr_path):
        st.error(f"❌ Path not found: `{zarr_path}`")
        return

    sample_id = Path(zarr_path).stem.replace(".zarr", "")
    st.caption(f"**Sample:** {sample_id}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    with st.spinner("Loading zarr data..."):
        ds = load_zarr_with_raw(zarr_path)

    st.session_state._base_ds = ds

    if "pipeline_cache" not in st.session_state:
        st.session_state.pipeline_cache = {}
    if "zoom_img_cache" not in st.session_state:
        st.session_state.zoom_img_cache = {}
        st.session_state.zoom_img_cache_max = 200

    # ------------------------------------------------------------------
    # Initialize thresholds
    # ------------------------------------------------------------------
    initial_thresholds = compute_initial_thresholds(ds, marker_threshold_dict)
    existing_thresholds = load_existing_thresholds(sample_id)
    if existing_thresholds is not None:
        st.success("✅ Threshold found — loaded saved thresholds for this sample.")
        st.session_state.thresholds_from_file = True
        for marker in MARKERS:
            if marker in existing_thresholds:
                initial_thresholds[marker] = existing_thresholds[marker]
    else:
        st.session_state.thresholds_from_file = False

    for marker in MARKERS:
        key = f"marker_thresh_{marker}"
        if key not in st.session_state:
            default_val = initial_thresholds.get(marker, 0)
            st.session_state[key] = max(0, min(255, default_val))

    thresholds = {m: st.session_state[f"marker_thresh_{m}"] for m in MARKERS}
    st.session_state.thresholds = thresholds

    y_start, y_end, x_start, x_end = 0, ds.dims["y"], 0, ds.dims["x"]
    st.session_state._zoom_y = (y_start, y_end)
    st.session_state._zoom_x = (x_start, x_end)

    # ------------------------------------------------------------------
    # Pipeline execution (shared by top + bottom buttons)
    # ------------------------------------------------------------------

    def _execute_pipeline():
        try:
            with st.spinner("Running cell type prediction pipeline..."):
                ds_processed = _get_pipeline_cached(thresholds)
                st.session_state.ds_processed = ds_processed
                st.session_state.pipeline_has_run = True
                st.session_state.pipeline_executed = True

            thresh_key = _make_thresh_key(thresholds)
            if st.session_state.get("_rendered_thresh_key") != thresh_key:
                marker_list = list(MARKERS)
                detail_pngs = {}
                total = len(marker_list) + 1

                progress_bar = st.progress(0, text="Rendering overview...")
                overview_png = _render_overview_raw(ds_processed, y_start, y_end, x_start, x_end)

                with ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_marker = {}
                    for marker in marker_list:
                        future = executor.submit(
                            _render_detail_raw, ds_processed,
                            marker, all_marker_colors[marker], ct_marker_dict[marker],
                            thresholds[marker], y_start, y_end, x_start, x_end
                        )
                        future_to_marker[future] = marker

                    for i, future in enumerate(as_completed(future_to_marker)):
                        marker = future_to_marker[future]
                        detail_pngs[marker] = future.result()
                        progress_bar.progress(
                            (i + 2) / total,
                            text=f"Rendering {marker}... ({i + 1}/{len(marker_list)} markers)"
                        )

                progress_bar.progress(1.0, text="Rendering complete!")
                st.session_state.rendered_overview_png = overview_png
                st.session_state.rendered_detail_pngs = detail_pngs
                st.session_state._rendered_thresh_key = thresh_key
                progress_bar.empty()

            st.rerun()
        except Exception as e:
            st.error(f"❌ Error running pipeline: {e}")
            st.exception(e)

    if run_top:
        _execute_pipeline()

    # ------------------------------------------------------------------
    # Log saving helper
    # ------------------------------------------------------------------
    def _save_logs():
        os.makedirs(threshold_csv_dir, exist_ok=True)
        df_save = pd.DataFrame({
            "sample_id": sample_id,
            "channel": list(thresholds.keys()),
            "threshold": list(thresholds.values()),
        })
        csv_path = f"{threshold_csv_dir}/{sample_id}.csv"
        df_save.to_csv(csv_path)

        os.makedirs("logs", exist_ok=True)
        log_path = f"logs/{sample_id}.csv"
        log_rows = []
        for marker in MARKERS:
            marker_img = ds.pp[marker]["_raw_image"].values
            flat = marker_img.ravel()
            pct_vals = np.percentile(flat, [25, 50, 75, 90, 95, 99]).tolist()
            quantile_fraction = marker_threshold_dict.get(marker, None)
            intensity_at_quantile = int(np.percentile(flat, 100 * quantile_fraction)) if quantile_fraction is not None else None
            nonzero_mask = flat > 0
            n_nonzero = int(np.sum(nonzero_mask))
            celltype = ct_marker_dict[marker]
            initial_thresh = initial_thresholds.get(marker, None)
            log_rows.append({
                "sample_id": sample_id, "channel": marker, "celltype": celltype,
                "n_cells_total": int(ds.dims["cells"]),
                "img_dims_y": int(ds.dims["y"]), "img_dims_x": int(ds.dims["x"]),
                "pixels_total": int(ds.dims["y"] * ds.dims["x"]),
                "mean_intensity_raw": round(float(np.mean(flat)), 2),
                "std_intensity_raw": round(float(np.std(flat)), 2),
                "min_intensity_raw": round(float(np.min(flat)), 2),
                "max_intensity_raw": round(float(np.max(flat)), 2),
                "median_intensity_raw": round(pct_vals[1], 2),
                "pct_25_raw": round(pct_vals[0], 2),
                "pct_75_raw": round(pct_vals[2], 2),
                "pct_90_raw": round(pct_vals[3], 2),
                "pct_95_raw": round(pct_vals[4], 2),
                "pct_99_raw": round(pct_vals[5], 2),
                "n_nonzero_pixels": n_nonzero,
                "fraction_nonzero": round(float(n_nonzero / len(flat)), 6),
                "mean_nonzero_raw": round(float(np.mean(flat[nonzero_mask])) if n_nonzero > 0 else 0.0, 2),
                "config_quantile_fraction": quantile_fraction,
                "config_quantile_percentile": f"{100 * quantile_fraction:.0f}%" if quantile_fraction else "N/A",
                "intensity_at_config_percentile": intensity_at_quantile,
                "initial_threshold": initial_thresh,
                "current_threshold": thresholds[marker],
                "threshold_delta": thresholds[marker] - initial_thresh if initial_thresh is not None else None,
                "pct_pixels_above_threshold": round(float(np.mean(flat > thresholds[marker])), 6),
                "timestamp": datetime.now().isoformat(),
            })
        pd.DataFrame(log_rows).to_csv(log_path, index=False)

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------
    if st.session_state.get("pipeline_has_run") and st.session_state.get("ds_processed") is not None:
        overview_png = st.session_state.get("rendered_overview_png")

        overview_title = "📊 Overview (Thresholded from Saved File)" if st.session_state.get("thresholds_from_file") else "📊 Overview"
        st.subheader(overview_title)
        if overview_png is not None:
            st.image(overview_png)

        st.subheader("🔬 Per-Marker Detail")
        st.markdown(
            "Adjust the threshold slider **above** each image to see how "
            "the prediction changes for that marker only."
        )

        for marker in MARKERS:
            try:
                color = all_marker_colors[marker]
                celltype = ct_marker_dict[marker]
                st.markdown(f"**{marker} → {celltype}**")
                render_marker_fragment(marker, color, celltype)
            except Exception as e:
                st.error(f"Error: {marker} — {e}")

    else:
        if zarr_path:
            st.info(
                "Adjust the sliders above each marker image and click **Run Pipeline** "
                "to see results."
            )
            st.subheader("📂 Data Preview")
            st.markdown(
                f"Loaded zarr dataset with **{len(ds.coords['channels'])} channels**, "
                f"**{ds.dims['y']}×{ds.dims['x']} pixels**, "
                f"and **{ds.dims['cells']} segmented cells**."
            )
            st.markdown(
                "**Markers used for cell type prediction:** "
                + ", ".join([f"`{m}`" for m in MARKERS])
            )

    # ------------------------------------------------------------------
    # Bottom buttons (only after first execution)
    # ------------------------------------------------------------------
    if st.session_state.get("pipeline_executed"):
        st.divider()
        col_b1, col_b2 = st.columns([1, 1])
        run_bottom = col_b1.button(run_label, type="primary", use_container_width=True, key="run_bottom")
        if run_bottom:
            _execute_pipeline()
        if col_b2.button("💾 Save Threshold & Log", use_container_width=True, key="save_bottom"):
            _save_logs()
            st.success("✅ Threshold & Logs Saved.")
            st.cache_data.clear()


if __name__ == "__main__":
    main()
