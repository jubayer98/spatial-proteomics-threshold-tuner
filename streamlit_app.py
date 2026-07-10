"""
Spatial Proteomics Threshold Tuner - Streamlit App
===================================================
Interactive tool to tune thresholds for marker-based cell type prediction
from multiplexed imaging data (zarr format).

Usage:
    streamlit run streamlit_app.py
"""

import os
import sys
import io
import base64
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import xarray as xr
import spatialproteomics as sp
from streamlit_image_zoom import image_zoom

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spatial Proteomics Threshold Tuner",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for black background
st.markdown(
    """
    <style>
    .stApp {
        background-color: black;
        color: white;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp li, .stApp span, .stApp div:not([data-testid]) {
        color: white;
    }
    .stApp .stMarkdown, .stApp .stText {
        color: white;
    }
    .stApp [data-testid="stSidebar"] {
        background-color: #1e1e1e;
    }
    .stApp [data-testid="stSidebar"] * {
        color: white;
    }
    .stApp [data-testid="stSidebar"] .stMarkdown, 
    .stApp [data-testid="stSidebar"] .stText {
        color: white;
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
def load_zarr_data(zarr_path):
    """Load the full zarr dataset."""
    ds = xr.open_zarr(zarr_path)
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


def crop_ds(ds, y_start, y_end, x_start, x_end):
    """Crop dataset to the given region."""
    return ds.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))


def fig_to_rgb(fig):
    """Convert a matplotlib figure to an RGB numpy array."""
    fig.canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, facecolor=fig.get_facecolor())
    buf.seek(0)
    from PIL import Image
    img = Image.open(buf)
    return np.array(img)


def plot_overview(ds):
    """
    Create the 4-panel overview figure as a static matplotlib figure.
    """
    ncols = 4
    scaling = 5
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
    scaling = 4
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


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.title("🧬 Spatial Proteomics Threshold Tuner")
    st.markdown(
        """
        Interactive tool for tuning marker thresholds used in cell type prediction.
        Adjust the sliders for each marker and see how the cell type predictions change.
        """
    )

    # ---- Sidebar ----
    st.sidebar.header("⚙️ Controls")

    # ---- Data source ----
    st.sidebar.subheader("📁 Data Source")

    # Use a counter to force widget recreation on clear
    if "clear_counter" not in st.session_state:
        st.session_state.clear_counter = 0

    zarr_path = st.sidebar.text_input(
        "Zarr file path",
        value="",
        key=f"zarr_path_input_{st.session_state.clear_counter}",
        placeholder="e.g. /path/to/data.zarr",
        help="Path to the zarr dataset file",
    )

    # Clear data button — resets everything to initial state
    if st.sidebar.button("🗑️ Clear Data", use_container_width=True):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.clear_counter += 1
        st.rerun()

    # Check if zarr path exists
    if not zarr_path:
        st.info("👈 Enter a valid zarr file path in the sidebar to get started.")
        return
    if not os.path.exists(zarr_path):
        st.sidebar.error(f"❌ Zarr path not found: `{zarr_path}`")
        return

    # Sample info
    sample_id = Path(zarr_path).stem.replace(".zarr", "")
    st.sidebar.info(f"**Sample:** {sample_id}")

    # ---- Load data ----
    with st.spinner("Loading zarr data... This may take a moment."):
        ds = load_zarr_data(zarr_path)

    # Add raw image layer (copy of original intensities)
    ds = ds.pp.add_layer(
        ds["_image"].values.copy(), key_added="_raw_image"
    )

    # Compute initial thresholds
    initial_thresholds = compute_initial_thresholds(ds, marker_threshold_dict)

    # Check for existing saved thresholds
    existing_thresholds = load_existing_thresholds(sample_id)
    if existing_thresholds is not None:
        st.sidebar.success(
            f"✅ Loaded existing thresholds from `{threshold_csv_dir}/{sample_id}.csv`"
        )
        # Use existing thresholds as defaults, fall back to computed
        for marker in MARKERS:
            if marker in existing_thresholds:
                initial_thresholds[marker] = existing_thresholds[marker]

    # ---- Threshold sliders ----
    st.sidebar.subheader("Marker Thresholds")
    st.sidebar.markdown(
        "Adjust the intensity threshold for each marker (0–255 for 8-bit images)."
    )

    thresholds = {}
    for marker in MARKERS:
        slider_key = f"sidebar_thresh_{marker}"
        if slider_key not in st.session_state:
            default_val = initial_thresholds.get(marker, 0)
            default_val = max(0, min(255, default_val))
            st.session_state[slider_key] = default_val

        thresholds[marker] = st.sidebar.slider(
            f"{marker} → {ct_marker_dict[marker]}",
            min_value=0,
            max_value=255,
            key=slider_key,
            help=f"Threshold for {marker} (cell type: {ct_marker_dict[marker]})",
        )

    # No sidebar zoom controls — each marker has its own interactive zoom expander
    y_start, y_end, x_start, x_end = 0, ds.dims["y"], 0, ds.dims["x"]

    # ---- Action buttons ----
    col1, col2 = st.sidebar.columns(2)
    run_button = col1.button("🪂 Pipeline", type="primary", use_container_width=True)
    # Show "Update" if thresholds already exist, otherwise "Save"
    thresholds_exist = existing_thresholds is not None
    save_label = "🔄 Thresholds" if thresholds_exist else "💾 Thresholds"
    save_button = col2.button(save_label, use_container_width=True)

    # ---- Save thresholds ----
    if save_button:
        os.makedirs(threshold_csv_dir, exist_ok=True)
        df_save = pd.DataFrame(
            {
                "sample_id": sample_id,
                "channel": list(thresholds.keys()),
                "threshold": list(thresholds.values()),
            }
        )
        csv_path = f"{threshold_csv_dir}/{sample_id}.csv"
        df_save.to_csv(csv_path)
        st.sidebar.success(f"✅ Thresholds saved to `{csv_path}`")

        # ---- Save log file for this sample ----
        os.makedirs("logs", exist_ok=True)
        log_path = f"logs/{sample_id}.csv"

        # Compute global statistics shared across all markers
        n_cells_total = int(ds.dims["cells"])
        img_dims_y = int(ds.dims["y"])
        img_dims_x = int(ds.dims["x"])

        log_rows = []
        for marker in MARKERS:
            quantile_fraction = marker_threshold_dict.get(marker, None)
            marker_img = ds.pp[marker]["_raw_image"].values.astype(np.float64)
            flat = marker_img.ravel()

            # Basic statistics
            mean_raw = float(np.mean(flat))
            std_raw = float(np.std(flat))
            min_raw = float(np.min(flat))
            max_raw = float(np.max(flat))
            median_raw = float(np.median(flat))
            pct_25 = float(np.percentile(flat, 25))
            pct_75 = float(np.percentile(flat, 75))
            pct_90 = float(np.percentile(flat, 90))
            pct_95 = float(np.percentile(flat, 95))
            pct_99 = float(np.percentile(flat, 99))

            # Intensity at the config-specified quantile
            intensity_at_quantile = int(np.percentile(flat, 100 * quantile_fraction)) if quantile_fraction is not None else None
            pct_config = f"{100 * quantile_fraction:.0f}%" if quantile_fraction is not None else "N/A"

            # Nonzero statistics
            nonzero_mask = flat > 0
            n_nonzero = int(np.sum(nonzero_mask))
            frac_nonzero = float(n_nonzero / len(flat))
            mean_nonzero = float(np.mean(flat[nonzero_mask])) if n_nonzero > 0 else 0.0
            pct_above_threshold = float(np.mean(flat > thresholds[marker]))

            # Initial threshold (config-based, before user tuning)
            initial_thresh = initial_thresholds.get(marker, None)
            threshold_delta = thresholds[marker] - initial_thresh if initial_thresh is not None else None

            celltype = ct_marker_dict[marker]
            log_rows.append({
                "sample_id": sample_id,
                "channel": marker,
                "celltype": celltype,
                "n_cells_total": n_cells_total,
                "img_dims_y": img_dims_y,
                "img_dims_x": img_dims_x,
                "pixels_total": int(img_dims_y * img_dims_x),
                "mean_intensity_raw": round(mean_raw, 2),
                "std_intensity_raw": round(std_raw, 2),
                "min_intensity_raw": round(min_raw, 2),
                "max_intensity_raw": round(max_raw, 2),
                "median_intensity_raw": round(median_raw, 2),
                "pct_25_raw": round(pct_25, 2),
                "pct_75_raw": round(pct_75, 2),
                "pct_90_raw": round(pct_90, 2),
                "pct_95_raw": round(pct_95, 2),
                "pct_99_raw": round(pct_99, 2),
                "n_nonzero_pixels": n_nonzero,
                "fraction_nonzero": round(frac_nonzero, 6),
                "mean_nonzero_raw": round(mean_nonzero, 2),
                "config_quantile_fraction": quantile_fraction,
                "config_quantile_percentile": pct_config,
                "intensity_at_config_percentile": intensity_at_quantile,
                "initial_threshold": initial_thresh,
                "current_threshold": thresholds[marker],
                "threshold_delta": threshold_delta,
                "pct_pixels_above_threshold": round(pct_above_threshold, 6),
                "timestamp": datetime.now().isoformat(),
            })

        df_log = pd.DataFrame(log_rows)
        df_log.to_csv(log_path, index=False)
        st.sidebar.success(f"📋 Log saved to `{log_path}`")

        # Clear the cache so next load picks up new file
        st.cache_data.clear()

    # ---- Run pipeline ----
    if run_button:
        with st.spinner("Running cell type prediction pipeline..."):
            try:
                ds_processed = run_pipeline(ds, thresholds)
                # Store in session state for real-time zoom
                st.session_state.ds_processed = ds_processed
                st.session_state.pipeline_has_run = True
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error running pipeline: {e}")
                st.exception(e)

    # ---- Display results (from session state, supports real-time zoom) ----
    if st.session_state.get("pipeline_has_run") and st.session_state.get("ds_processed") is not None:
        ds_processed = st.session_state.ds_processed

        # Crop the dataset for zoom
        ds_cropped = crop_ds(ds_processed, y_start, y_end, x_start, x_end)

        # ---- Overview section ----
        st.subheader("📊 Overview")
        st.markdown(
            "Four-panel view showing segmentation, raw composite, "
            "thresholded composite, and predicted cell types."
        )
        fig_overview = plot_overview(ds_cropped)
        st.pyplot(fig_overview)
        plt.close(fig_overview)

        # ---- Per-marker detail section ----
        st.subheader("🔬 Per-Marker Detail")
        st.markdown(
            "For each marker, view the raw signal, thresholded signal, "
            "and the corresponding predicted cell type."
        )

        for marker in MARKERS:
            try:
                color = all_marker_colors[marker]
                celltype = ct_marker_dict[marker]
                threshold_value = thresholds[marker]
                fig_detail = plot_marker_detail(
                    ds_cropped, marker, color, celltype, threshold_value
                )
                st.pyplot(fig_detail)
                plt.close(fig_detail)

                # Interactive zoom expander for each marker
                with st.expander(f"🔍 Interactive Zoom: {marker} → {celltype}"):
                    st.markdown(
                        f"**{marker}** — adjust threshold and explore with synchronized interactive zoom. "
                        "🖱️ Click/drag on any image to zoom/pan — all three images move together."
                    )

                    # Local threshold slider for this marker
                    local_thresh = st.slider(
                        f"Threshold for {marker}",
                        min_value=0, max_value=255,
                        value=threshold_value,
                        key=f"zoom_thresh_{marker}",
                    )

                    # Re-run pipeline with updated threshold for this marker
                    local_thresholds = thresholds.copy()
                    local_thresholds[marker] = local_thresh
                    with st.spinner(f"Re-running pipeline for {marker}..."):
                        try:
                            ds_local = run_pipeline(ds, local_thresholds)
                            ds_local_cropped = crop_ds(ds_local, y_start, y_end, x_start, x_end)

                            # Render each panel as a separate image (no titles)
                            def render_single_panel(ds_data, channel, panel_color, layer_key, render_segmentation=False, render_labels=False, ct=None):
                                fig, ax = plt.subplots(1, 1, figsize=(4, 4), facecolor="black")
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
                                rgb = fig_to_rgb(fig)
                                plt.close(fig)
                                return rgb

                            img_raw = render_single_panel(ds_local_cropped, marker, color, "_raw_image")
                            img_thresh = render_single_panel(ds_local_cropped, marker, color, "_thresholded_image")
                            img_pred = render_single_panel(ds_local_cropped, marker, color, "_thresholded_image",
                                                           render_labels=True, ct=celltype)

                            # Convert all images to base64
                            def img_to_b64(rgb):
                                from PIL import Image
                                pil_img = Image.fromarray(rgb)
                                # Convert RGBA to RGB if needed (JPEG doesn't support alpha)
                                if pil_img.mode == 'RGBA':
                                    pil_img = pil_img.convert('RGB')
                                buf = io.BytesIO()
                                pil_img.save(buf, format="JPEG", subsampling=0, quality=95)
                                return base64.b64encode(buf.getvalue()).decode()

                            b64_raw = img_to_b64(img_raw)
                            b64_thresh = img_to_b64(img_thresh)
                            b64_pred = img_to_b64(img_pred)

                            # Custom HTML with synchronized zoom across 3 images
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
                                    <img id="img-{marker}-0" src="data:image/jpeg;base64,{b64_raw}" data-scale="1" data-ox="0.5" data-oy="0.5">
                                </div>
                                <div class="zoom-cell-{marker}">
                                    <img id="img-{marker}-1" src="data:image/jpeg;base64,{b64_thresh}" data-scale="1" data-ox="0.5" data-oy="0.5">
                                </div>
                                <div class="zoom-cell-{marker}">
                                    <img id="img-{marker}-2" src="data:image/jpeg;base64,{b64_pred}" data-scale="1" data-ox="0.5" data-oy="0.5">
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
                                        // Click to zoom in at click position
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
                            st.components.v1.html(html, height=450)

                        except Exception as e:
                            st.error(f"Error in zoom view for {marker}: {e}")

            except Exception as e:
                st.error(f"Error plotting {marker}: {e}")
    else:
        st.info(
            "👈 Adjust the threshold sliders in the sidebar and click **Run Pipeline** "
            "to see the results."
        )

        # Show a preview of the data
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


if __name__ == "__main__":
    main()
