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
import warnings
from pathlib import Path

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


def plot_overview(ds):
    """
    Create the 4-panel overview figure:
    - DAPI Segmentation (DAPI + segmentation overlay)
    - Raw composite (5 major channels: PAX5, CD3, CD11c, CD11b, CD68)
    - Thresholded composite (same 5 channels)
    - Predicted cell types (5 major types: BCell, TCell, Dendritic, Myeloid, Macro)
    """
    ncols = 4
    scaling = 5
    fig, ax = plt.subplots(1, ncols, figsize=(ncols * scaling, scaling),
                           facecolor="black")
    ax = ax.flatten()
    for axis in ax:
        axis.set_facecolor("black")

    # Panel 1: DAPI Segmentation
    ds.pp["DAPI"].pl.colorize("gold").pl.show(
        render_segmentation=True, ax=ax[0]
    )
    ax[0].set_title("DAPI Segmentation", fontsize=14, fontweight="bold", color="white")

    # Panel 2: Raw composite (5 major channels)
    overview_colors = [all_marker_colors[ch] for ch in OVERVIEW_CHANNELS]
    ds.pp[OVERVIEW_CHANNELS].pl.colorize(
        overview_colors, layer_key="_raw_image"
    ).pl.show(ax=ax[1])
    ax[1].set_title("RAW", fontsize=14, fontweight="bold", color="white")

    # Panel 3: Thresholded composite (same 5 channels)
    ds.pp[OVERVIEW_CHANNELS].pl.colorize(
        overview_colors, layer_key="_thresholded_image"
    ).pl.show(ax=ax[2])
    ax[2].set_title("Threshold", fontsize=14, fontweight="bold", color="white")

    # Panel 4: Predicted cell types (filtered to 5 major types)
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
    Create a 4-panel detail view for a single marker:
    - Marker raw + segmentation overlay
    - Marker raw
    - Marker thresholded
    - Associated cell type labels
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
        default_val = initial_thresholds.get(marker, 0)
        # Clamp to 0-255
        default_val = max(0, min(255, default_val))
        thresholds[marker] = st.sidebar.slider(
            f"{marker} → {ct_marker_dict[marker]}",
            min_value=0,
            max_value=255,
            value=default_val,
            help=f"Threshold for {marker} (cell type: {ct_marker_dict[marker]})",
        )

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
        # Clear the cache so next load picks up new file
        st.cache_data.clear()

    # ---- Run pipeline ----
    if run_button:
        with st.spinner("Running cell type prediction pipeline..."):
            try:
                ds_processed = run_pipeline(ds, thresholds)

                # ---- Overview section ----
                st.subheader("📊 Overview")
                st.markdown(
                    "Four-panel view showing segmentation, raw composite, "
                    "thresholded composite, and predicted cell types."
                )
                fig_overview = plot_overview(ds_processed)
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
                            ds_processed, marker, color, celltype, threshold_value
                        )
                        st.pyplot(fig_detail)
                        plt.close(fig_detail)
                    except Exception as e:
                        st.error(f"Error plotting {marker}: {e}")


            except Exception as e:
                st.error(f"❌ Error running pipeline: {e}")
                st.exception(e)
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
