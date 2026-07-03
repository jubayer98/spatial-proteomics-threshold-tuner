import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from skimage.draw import polygon
import ast

# Import our modules
from config import load_config
from data_loader import find_image_data, find_segmentation_labels
from processing import (
    compute_thresholded_masks,
    composite_masks,
    composite_raw,
    recompute_predictions,
    save_thresholds_to_file
)
from visualization import (
    plot_raw_image,
    plot_thresholded_intensity,
    plot_predicted_image,
    plot_composite_image
)

st.set_page_config(page_title="SP Tuner", layout="wide")
st.title("🔬 Spatial Proteomics – Thresholds Tuner")

# ---------- session state initialisation ----------
if "thresholds" not in st.session_state:
    st.session_state.thresholds = {}
if "cell_means" not in st.session_state:
    st.session_state.cell_means = {}
if "predicted_masks" not in st.session_state:
    st.session_state.predicted_masks = {}
if "labels" not in st.session_state:
    st.session_state.labels = None
if "image_data" not in st.session_state:
    st.session_state.image_data = {}
if "thresholds_saved" not in st.session_state:
    st.session_state.thresholds_saved = False
if "threshold_file_path" not in st.session_state:
    st.session_state.threshold_file_path = None
if "artefact_mask" not in st.session_state:
    st.session_state.artefact_mask = None  # optional binary mask

# ---------- sidebar: Zarr path ----------
zarr_path = st.sidebar.text_input("📂 Zarr file location", value="")

if not zarr_path:
    st.info("👈 Enter a Zarr file path in the sidebar to get started.")
    st.stop()

# ---------- Load config ----------
config = load_config()
if not config:
    st.sidebar.warning("⚠️ config.yml not found. Please place it in the same directory.")
    st.stop()
st.sidebar.info("💡 Config file loaded.")

marker_colors = config.get('all_marker_colors', {})
ct_marker_dict = config.get('ct_marker_dict', {})
threshold_output_dir = config.get('threshold_output_dir', './thresholds')
if not marker_colors:
    st.warning("No markers found under `all_marker_colors` in config.yml. Nothing to display.")
    st.stop()

st.session_state.marker_colors = marker_colors
st.session_state.ct_marker_dict = ct_marker_dict

# ---------- Derive output filename ----------
zarr_filename = Path(zarr_path).name
if zarr_filename.startswith("segmented_"):
    base_name = zarr_filename.replace("segmented_", "threshold_").replace(".zarr", ".txt")
else:
    base_name = Path(zarr_filename).stem + "_threshold.txt"
threshold_file_path = Path(threshold_output_dir) / base_name

# ---------- Load dataset ----------
try:
    ds = xr.open_zarr(zarr_path)
    st.sidebar.success("✅ Dataset loaded")
except Exception as e:
    st.error(f"Failed to load Zarr: {e}")
    st.stop()

# ---------- Extract image data ----------
try:
    image_data, channel_dim, channel_names = find_image_data(ds, marker_colors)
except ValueError as e:
    st.error(str(e))
    st.stop()

if not image_data:
    st.error("No valid 2D images found for any marker in config.")
    st.stop()
st.session_state.image_data = image_data

# ---------- Extract segmentation labels ----------
labels = find_segmentation_labels(ds)
if labels is not None:
    st.session_state.labels = labels
    st.sidebar.success("✅ Segmentation mask found.")
else:
    st.sidebar.warning("⚠️ No segmentation mask found. Predicted images disabled.")
    st.session_state.labels = None

# ---------- Artefact handling (optional) ----------
apply_artefacts = st.sidebar.checkbox("Apply artefact masks", value=False)
if apply_artefacts:
    artefact_file = st.sidebar.file_uploader("Upload artefact CSV", type=["csv"])
    if artefact_file is not None:
        try:
            artefact_df = pd.read_csv(artefact_file)
            # Expected columns: sample, polygon   (polygon as string representation of list of tuples)
            # Find rows for current sample
            sample_id = zarr_filename.replace('.zarr', '')
            sample_rows = artefact_df[artefact_df['sample'] == sample_id]
            if not sample_rows.empty:
                img_shape = (ds.sizes["y"], ds.sizes["x"])
                full_mask = np.ones(img_shape, dtype=np.uint8)
                for _, row in sample_rows.iterrows():
                    polygon_coords = ast.literal_eval(row['polygon'])
                    poly = Polygon(polygon_coords)
                    if not poly.is_valid:
                        continue
                    x, y = zip(*polygon_coords)
                    y_flipped = [img_shape[0] - yi for yi in y]
                    rr, cc = polygon(y_flipped, x, img_shape)
                    mask = np.ones(img_shape, dtype=np.uint8)
                    mask[rr, cc] = 0
                    full_mask = np.minimum(full_mask, mask)
                st.session_state.artefact_mask = full_mask
                st.sidebar.success(f"Loaded {len(sample_rows)} artefact polygon(s)")
            else:
                st.sidebar.info("No artefacts found for this sample.")
                st.session_state.artefact_mask = None
        except Exception as e:
            st.sidebar.error(f"Failed to parse artefact CSV: {e}")
            st.session_state.artefact_mask = None
    else:
        st.session_state.artefact_mask = None
else:
    st.session_state.artefact_mask = None

# ---------- Apply artefact mask to labels ----------
if st.session_state.artefact_mask is not None and labels is not None:
    # Zero out labels that overlap with artefact regions
    labels = labels.copy()
    labels[st.session_state.artefact_mask == 0] = 0
    st.session_state.labels = labels  # update with masked labels
    st.sidebar.success("Artefact regions excluded from segmentation.")

# ---------- Load existing thresholds ----------
if threshold_file_path.exists():
    try:
        saved_thr = pd.read_csv(threshold_file_path)
        # saved_thr columns: channel_id, threshold_value
        for _, row in saved_thr.iterrows():
            marker = row['channel_id']
            val = float(row['threshold_value'])
            if marker in image_data:
                st.session_state.thresholds[marker] = val
        st.session_state.thresholds_saved = True
        st.session_state.threshold_file_path = str(threshold_file_path)
    except Exception:
        pass

# ---------- Initialize missing thresholds from config (if not loaded from file) ----------
marker_threshold_quantiles = config.get('marker_threshold_dict', {})
for marker, img in image_data.items():
    if marker not in st.session_state.thresholds:
        if marker in marker_threshold_quantiles:
            fraction = marker_threshold_quantiles[marker]
            threshold_int = int(np.percentile(img, 100 * fraction))
            st.session_state.thresholds[marker] = threshold_int
        else:
            st.session_state.thresholds[marker] = float(np.mean(img))

# ---------- Save / Update button ----------
save_label = "🔄 Update Thresholds" if st.session_state.thresholds_saved else "💾 Save Thresholds"
if st.sidebar.button(save_label):
    if not st.session_state.thresholds:
        st.sidebar.warning("No thresholds available. Adjust sliders first.")
    else:
        success = save_thresholds_to_file(
            st.session_state.thresholds,
            marker_colors,
            st.session_state.image_data.keys(),
            threshold_output_dir,
            base_name
        )
        if success:
            st.sidebar.success(f"Thresholds saved to {threshold_file_path}")
            st.session_state.thresholds_saved = True
            st.rerun()

# ---------- Initialise predictions ----------
if st.session_state.labels is not None:
    predicted_masks, cell_means = recompute_predictions(
        marker_colors,
        ct_marker_dict,
        st.session_state.labels,
        image_data,
        st.session_state.thresholds
    )
    st.session_state.predicted_masks = predicted_masks
    st.session_state.cell_means = cell_means

# ---------- Toggle segmentation outlines ----------
show_segmentation = st.sidebar.checkbox("Show segmentation outlines", value=False)
if show_segmentation and st.session_state.labels is None:
    st.sidebar.warning("No segmentation mask available – outlines cannot be shown.")
    show_segmentation = False

# ---------- Display each marker ----------
for marker, color_hex in marker_colors.items():
    if marker not in image_data:
        st.warning(f"Marker '{marker}' not in dataset. Skipping.")
        continue

    raw_img = image_data[marker]
    vmin = int(np.min(raw_img))
    vmax = int(np.max(raw_img))

    st.markdown(f"#### <font color='{color_hex}'>{marker}</font>", unsafe_allow_html=True)

    current_thresh = st.session_state.thresholds.get(marker, np.mean(raw_img))
    if not isinstance(current_thresh, int):
        current_thresh = int(current_thresh)

    new_thresh = st.slider(
        f"Threshold for {marker}",
        min_value=vmin,
        max_value=vmax,
        value=current_thresh,
        step=1,
        key=f"slider_{marker}"
    )
    if new_thresh != current_thresh:
        st.session_state.thresholds[marker] = new_thresh
        if st.session_state.labels is not None:
            pred_masks, cell_means = recompute_predictions(
                marker_colors,
                ct_marker_dict,
                st.session_state.labels,
                image_data,
                st.session_state.thresholds
            )
            st.session_state.predicted_masks = pred_masks
            st.session_state.cell_means = cell_means

    thresh_mask = raw_img > new_thresh
    predicted_mask = st.session_state.predicted_masks.get(marker, None)

    boundaries = None
    if show_segmentation and st.session_state.labels is not None:
        boundaries = find_boundaries(st.session_state.labels, mode='inner')

    col1, col2, col3 = st.columns(3)

    with col1:
        fig = plot_raw_image(raw_img, color_hex, "Raw", boundaries)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig = plot_thresholded_intensity(raw_img, new_thresh, color_hex,
                                     f"Threshold ({new_thresh:.2f})", boundaries)
        st.pyplot(fig)
        plt.close(fig)

    with col3:
        if predicted_mask is not None:
            celltype = ct_marker_dict.get(marker, "")
            fig = plot_predicted_image(predicted_mask, color_hex, f"Predicted ({celltype})", boundaries)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No prediction (segmentation missing)")

# ---------- Sidebar: show current thresholds ----------
with st.sidebar.expander("Current thresholds"):
    for m, t in st.session_state.thresholds.items():
        st.write(f"{m}: {t:.2f}")