import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import yaml
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
import os

st.set_page_config(page_title="Spatial Proteomics Viewer", layout="wide")
st.title("🔬 Spatial Proteomics – Raw, Threshold (slider) & Predicted")

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

# ---------- helper to recompute predictions ----------
def recompute_all(updated_marker=None):
    marker_colors = st.session_state.get("marker_colors", {})
    ct_marker_dict = st.session_state.get("ct_marker_dict", {})
    labels = st.session_state.labels
    image_data = st.session_state.image_data
    thresholds = st.session_state.thresholds
    cell_means = st.session_state.cell_means

    if labels is None or not image_data:
        return

    # 1. Recompute per‑cell means for the updated marker (or all)
    if updated_marker is not None and updated_marker in image_data:
        raw = image_data[updated_marker]
        thresh_val = thresholds.get(updated_marker, np.mean(raw))
        thresh_img = raw.copy()
        thresh_img[thresh_img < thresh_val] = 0
        props = regionprops(labels, intensity_image=thresh_img)
        cell_means[updated_marker] = np.array([p.mean_intensity for p in props])
    else:
        for marker, raw in image_data.items():
            thresh_val = thresholds.get(marker, np.mean(raw))
            thresh_img = raw.copy()
            thresh_img[thresh_img < thresh_val] = 0
            props = regionprops(labels, intensity_image=thresh_img)
            cell_means[marker] = np.array([p.mean_intensity for p in props])

    # 2. Build matrix (cells x markers)
    markers_list = list(marker_colors.keys())
    available_markers = [m for m in markers_list if m in cell_means and cell_means[m].size > 0]
    if not available_markers:
        st.session_state.predicted_masks = {}
        return

    X = np.column_stack([cell_means[m] for m in available_markers])
    X_trans = np.arcsinh(X)
    argmax_idx = np.argmax(X_trans, axis=1)
    cell_types = [ct_marker_dict.get(available_markers[idx], "Unknown") for idx in argmax_idx]

    predicted_masks = {}
    for marker in available_markers:
        celltype = ct_marker_dict.get(marker, None)
        if celltype is None:
            continue
        mask_cells = np.array([ct == celltype for ct in cell_types])
        cell_indices = np.where(mask_cells)[0]
        label_values = cell_indices + 1
        mask_2d = np.isin(labels, label_values)
        predicted_masks[marker] = mask_2d

    st.session_state.predicted_masks = predicted_masks

# ---------- helper to save thresholds ----------
def save_thresholds(output_dir, base_filename):
    """Write thresholds to a text file: channel_id,threshold_value."""
    markers = sorted(st.session_state.marker_colors.keys())
    # Only those that exist in image_data
    markers = [m for m in markers if m in st.session_state.image_data]
    if not markers:
        st.error("No markers available to save thresholds.")
        return False

    # Build file path
    out_path = Path(output_dir) / base_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(out_path, 'w') as f:
            f.write("channel_id,threshold_value\n")
            for marker in markers:
                th = st.session_state.thresholds.get(marker, np.nan)
                f.write(f"{marker},{th:.6f}\n")
        st.session_state.thresholds_saved = True
        st.session_state.threshold_file_path = str(out_path)
        return True
    except Exception as e:
        st.error(f"Failed to save thresholds: {e}")
        return False

# ---------- sidebar: Zarr path and controls ----------
zarr_path = st.sidebar.text_input("📂 Zarr file location", value="")

if zarr_path:
    # ---------- load config ----------
    config_path = Path(__file__).parent / 'config.yml'
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        st.sidebar.info("💡 Config file loaded.")
    else:
        st.sidebar.warning("⚠️ config.yml not found. Please place it in the same directory.")

    marker_colors = config.get('all_marker_colors', {})
    ct_marker_dict = config.get('ct_marker_dict', {})
    threshold_output_dir = config.get('threshold_output_dir', './thresholds')
    if not marker_colors:
        st.warning("No markers found under `all_marker_colors` in config.yml. Nothing to display.")
        st.stop()

    st.session_state.marker_colors = marker_colors
    st.session_state.ct_marker_dict = ct_marker_dict

    # Extract base filename for threshold output
    # e.g., "segmented_1-17.zarr" -> "threshold_1-17.txt"
    zarr_filename = Path(zarr_path).name
    if zarr_filename.startswith("segmented_"):
        base_name = zarr_filename.replace("segmented_", "threshold_").replace(".zarr", ".txt")
    else:
        base_name = Path(zarr_filename).stem + "_threshold.txt"
    threshold_file_path = Path(threshold_output_dir) / base_name

    # Check if file already exists -> set saved state to True
    if threshold_file_path.exists():
        st.session_state.thresholds_saved = True
        st.session_state.threshold_file_path = str(threshold_file_path)
    else:
        st.session_state.thresholds_saved = False
        st.session_state.threshold_file_path = None

    # ---------- Save / Update button ----------
    if st.session_state.thresholds_saved:
        save_label = "🔄 Update Thresholds"
    else:
        save_label = "💾 Save Thresholds"

    if st.sidebar.button(save_label):
        # Ensure we have thresholds for all markers
        if not st.session_state.thresholds:
            st.sidebar.warning("No thresholds available. Adjust sliders first.")
        else:
            success = save_thresholds(threshold_output_dir, base_name)
            if success:
                st.sidebar.success(f"Thresholds saved to {threshold_file_path}")
                st.session_state.thresholds_saved = True
                st.rerun()  # to update button text

    try:
        ds = xr.open_zarr(zarr_path)
        st.sidebar.success("✅ Dataset loaded")

        # ---------- Find image data ----------
        image_var = None
        channel_dim = None
        channel_candidates = ['channel', 'channels', 'Channel', 'Channels']
        spatial_candidates = ['x', 'y', 'X', 'Y', 'dim_0', 'dim_1']

        for var_name, var in ds.data_vars.items():
            dims = var.dims
            found_channel = None
            for cand in channel_candidates:
                if cand in dims:
                    found_channel = cand
                    break
            if found_channel is None:
                continue
            spatial = [d for d in dims if d in spatial_candidates]
            if len(spatial) >= 2:
                image_var = var
                channel_dim = found_channel
                break

        if image_var is None:
            for var_name, var in ds.data_vars.items():
                if len(var.dims) == 3:
                    dim_sizes = [var.sizes[d] for d in var.dims]
                    if max(dim_sizes) > 1:
                        image_var = var
                        for d in var.dims:
                            if var.sizes[d] > 1:
                                channel_dim = d
                                break
                        break

        if image_var is None:
            st.error("Could not find image data with a channel dimension and spatial dimensions.")
            st.stop()

        if channel_dim in image_var.coords:
            channel_names = [str(c) for c in image_var.coords[channel_dim].values]
        else:
            channel_names = [str(i) for i in range(image_var.sizes[channel_dim])]

        # ---------- load raw images for markers ----------
        image_data = {}
        for marker in marker_colors:
            if marker not in channel_names:
                continue
            try:
                if channel_dim in image_var.coords:
                    coords_as_str = [str(c) for c in image_var.coords[channel_dim].values]
                    idx = coords_as_str.index(marker)
                    img = image_var.isel({channel_dim: idx}).values
                else:
                    idx = channel_names.index(marker)
                    img = image_var.isel({channel_dim: idx}).values
                img = np.squeeze(img)
                if img.ndim == 2:
                    image_data[marker] = img
                else:
                    st.warning(f"Channel {marker} has shape {img.shape}, skipping.")
            except Exception as e:
                st.warning(f"Could not load {marker}: {e}")

        if not image_data:
            st.error("No valid 2D images found for any marker in config.")
            st.stop()

        st.session_state.image_data = image_data

        # ---------- Find segmentation labels ----------
        label_var = None
        for var_name, var in ds.data_vars.items():
            if len(var.dims) == 2 and var.dtype.kind in 'iu':
                label_var = var
                break
        if label_var is None:
            for var_name, var in ds.data_vars.items():
                if 'label' in var_name.lower() and len(var.dims) == 2:
                    label_var = var
                    break

        if label_var is not None:
            labels = label_var.values
            st.session_state.labels = labels
            st.sidebar.success("✅ Segmentation mask found.")
        else:
            st.sidebar.warning("⚠️ No segmentation mask found. Predicted images and outlines disabled.")
            st.session_state.labels = None

        # ---------- Toggle segmentation outlines ----------
        show_segmentation = st.sidebar.checkbox("Show segmentation outlines", value=False)
        if show_segmentation and st.session_state.labels is None:
            st.sidebar.warning("No segmentation mask available – outlines cannot be shown.")
            show_segmentation = False

        # ---------- Initialise thresholds and per‑cell means ----------
        for marker, img in image_data.items():
            if marker not in st.session_state.thresholds:
                st.session_state.thresholds[marker] = float(np.mean(img))

        if st.session_state.labels is not None:
            for marker in image_data:
                if marker not in st.session_state.thresholds:
                    st.session_state.thresholds[marker] = float(np.mean(image_data[marker]))
            recompute_all(updated_marker=None)

        # ---------- Display each marker with slider ----------
        st.markdown("---")
        st.markdown("### Adjust thresholds (drag slider) – updates instantly")

        for marker, color_hex in marker_colors.items():
            if marker not in image_data:
                st.warning(f"Marker '{marker}' not in dataset. Skipping.")
                continue

            raw_img = image_data[marker]
            vmin, vmax = float(np.min(raw_img)), float(np.max(raw_img))

            st.markdown(f"#### <font color='{color_hex}'>{marker}</font>", unsafe_allow_html=True)

            # Slider
            current_thresh = st.session_state.thresholds.get(marker, np.mean(raw_img))
            new_thresh = st.slider(
                f"Threshold for {marker}",
                min_value=vmin,
                max_value=vmax,
                value=current_thresh,
                step=(vmax - vmin) / 100.0,
                key=f"slider_{marker}"
            )
            # Update threshold if changed
            if new_thresh != current_thresh:
                st.session_state.thresholds[marker] = new_thresh
                recompute_all(marker)  # recompute predicted masks for this marker

            thresh_mask = raw_img > new_thresh
            predicted_mask = st.session_state.predicted_masks.get(marker, None)

            col1, col2, col3 = st.columns(3)

            # ---------- Prepare segmentation boundaries if enabled ----------
            boundaries = None
            if show_segmentation and st.session_state.labels is not None:
                boundaries = find_boundaries(st.session_state.labels, mode='inner')

            # ---------- Raw ----------
            with col1:
                fig, ax = plt.subplots(figsize=(5, 5))
                fig.patch.set_facecolor('black')
                ax.set_facecolor('black')
                # normalise raw
                p1, p99 = np.percentile(raw_img, (1, 99))
                raw_norm = np.clip((raw_img - p1) / (p99 - p1 + 1e-8), 0, 1)
                cmap = LinearSegmentedColormap.from_list('custom', ['black', color_hex])
                ax.imshow(raw_norm, cmap=cmap)
                if boundaries is not None:
                    # overlay boundaries in white
                    ax.imshow(boundaries, cmap='gray', alpha=0.5, interpolation='none')
                ax.set_title("Raw", color='white', fontsize=12)
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)

            # ---------- Thresholded ----------
            with col2:
                fig, ax = plt.subplots(figsize=(5, 5))
                fig.patch.set_facecolor('black')
                ax.set_facecolor('black')
                rgb = np.zeros((*thresh_mask.shape, 3), dtype=np.uint8)
                hex_color = color_hex.lstrip('#')
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                rgb[thresh_mask] = [r, g, b]
                ax.imshow(rgb)
                if boundaries is not None:
                    ax.imshow(boundaries, cmap='gray', alpha=0.5, interpolation='none')
                ax.set_title(f"Threshold ({new_thresh:.2f})", color='white', fontsize=12)
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)

            # ---------- Predicted ----------
            with col3:
                if predicted_mask is not None:
                    fig, ax = plt.subplots(figsize=(5, 5))
                    fig.patch.set_facecolor('black')
                    ax.set_facecolor('black')
                    rgb_pred = np.zeros((*predicted_mask.shape, 3), dtype=np.uint8)
                    rgb_pred[predicted_mask] = [r, g, b]
                    ax.imshow(rgb_pred)
                    if boundaries is not None:
                        ax.imshow(boundaries, cmap='gray', alpha=0.5, interpolation='none')
                    celltype = ct_marker_dict.get(marker, "")
                    ax.set_title(f"Predicted ({celltype})", color='white', fontsize=12)
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("No prediction (segmentation missing)")

        # ---------- show thresholds in sidebar ----------
        with st.sidebar.expander("Current thresholds"):
            for m, t in st.session_state.thresholds.items():
                st.write(f"{m}: {t:.2f}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("👈 Enter a Zarr file path in the sidebar to get started.")