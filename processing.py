import numpy as np
from matplotlib.colors import to_rgb
from skimage.measure import regionprops
from pathlib import Path

def compute_absolute_thresholds(image_data, fractional_thresholds, markers_to_skip=None):
    """
    Convert fractional thresholds (0..1) to absolute intensity values
    using percentiles of each marker's image.
    Returns a dict {marker: absolute_threshold}.
    """
    if markers_to_skip is None:
        markers_to_skip = ['DAPI', 'Dapi']
    abs_thresholds = {}
    for marker, img in image_data.items():
        if marker in markers_to_skip:
            continue
        if marker in fractional_thresholds:
            percentile = 100 * fractional_thresholds[marker]
            abs_thresholds[marker] = np.percentile(img, percentile)
        else:
            # Fallback: use mean (or you could raise a warning and use median)
            abs_thresholds[marker] = np.mean(img)
    return abs_thresholds

def compute_thresholded_masks(image_data, absolute_thresholds, markers_to_skip=None):
    """Return a dict of binary masks for each marker (above absolute threshold)."""
    if markers_to_skip is None:
        markers_to_skip = ['DAPI', 'Dapi']
    masks = {}
    for marker, img in image_data.items():
        if marker in markers_to_skip:
            continue
        th = absolute_thresholds.get(marker)
        if th is None:
            continue
        masks[marker] = img > th
    return masks

def composite_masks(masks_dict, colors_dict, markers_to_skip=None):
    """Create an RGB composite from binary masks."""
    if markers_to_skip is None:
        markers_to_skip = ['DAPI', 'Dapi']
    if not masks_dict:
        return None
    shape = next(iter(masks_dict.values())).shape
    composite = np.zeros((*shape, 3), dtype=np.float32)
    for marker, mask in masks_dict.items():
        if marker in markers_to_skip:
            continue
        color = to_rgb(colors_dict.get(marker, '#ffffff'))
        composite[mask] += np.array(color)
    return np.clip(composite, 0, 1)

def composite_raw(image_data, colors_dict, markers_to_skip=None, norm_percentiles=(1,99)):
    """Overlay multiple raw channels with their colors, each stretched independently."""
    if markers_to_skip is None:
        markers_to_skip = ['DAPI', 'Dapi']
    if not image_data:
        return None
    shape = next(iter(image_data.values())).shape
    composite = np.zeros((*shape, 3), dtype=np.float32)
    for marker, img in image_data.items():
        if marker in markers_to_skip or marker not in colors_dict:
            continue
        color = to_rgb(colors_dict[marker])
        p1, p99 = np.percentile(img, norm_percentiles)
        norm_img = np.clip((img - p1) / (p99 - p1 + 1e-8), 0, 1)
        composite += norm_img[..., np.newaxis] * np.array(color)
    return np.clip(composite, 0, 1)

def recompute_predictions(marker_colors, ct_marker_dict, labels, image_data,
                           absolute_thresholds=None, cell_mask=None):
    """
    Compute per-cell means from raw images, threshold them, and assign cell
    types by argmax of arcsinh-transformed, THRESHOLDED expression — matching
    the notebook pipeline (pp.threshold -> add_quantification -> arcsinh ->
    predict_cell_types_argmax).

    Parameters
    ----------
    marker_colors : dict
        marker -> color, used to determine marker ordering/inclusion.
    ct_marker_dict : dict
        marker -> cell type label.
    labels : 2D int array
        Segmentation label image (0 = background, 1..N = cell IDs).
    image_data : dict
        marker -> 2D raw intensity image.
    absolute_thresholds : dict, optional
        marker -> absolute intensity threshold. If a marker has no entry,
        thresholding falls back to raw (unthresholded) means for that marker.
        If None entirely, all markers use raw means (old behavior).
    cell_mask : 2D bool array, optional
        Same shape as `labels`. Pixels where this is False are excluded from
        quantification (e.g. artefact regions) before per-cell means are
        computed. Cells that end up with zero valid pixels are typed "Unknown".

    Returns
    -------
    predicted_masks : dict
        marker -> 2D boolean mask of cells assigned to that marker's cell type.
    cell_means : dict
        marker -> 1D array of per-cell mean intensities (thresholded where a
        threshold was supplied, raw otherwise). This is also what's used for
        cell typing, so it now matches what's plotted/reported.
    """
    if labels is None or not image_data:
        return {}, {}

    work_labels = labels
    if cell_mask is not None:
        # Zero out label IDs outside the valid (non-artefact) region so that
        # regionprops ignores those pixels entirely for masked-out cells.
        work_labels = np.where(cell_mask, labels, 0)

    # Compute per-cell means from raw intensities (no thresholding) — used as
    # a fallback for markers without a supplied threshold.
    cell_means_raw = {}
    for marker, raw in image_data.items():
        props = regionprops(work_labels, intensity_image=raw)
        cell_means_raw[marker] = np.array([p.mean_intensity for p in props])

    # Thresholded per-cell means (matches notebook's pp.threshold step,
    # applied BEFORE quantification/typing).
    cell_means = {}
    if absolute_thresholds is not None:
        for marker, raw in image_data.items():
            th = absolute_thresholds.get(marker)
            if th is None:
                cell_means[marker] = cell_means_raw[marker]
            else:
                thresh_img = raw.copy()
                thresh_img[thresh_img < th] = 0
                props = regionprops(work_labels, intensity_image=thresh_img)
                cell_means[marker] = np.array([p.mean_intensity for p in props])
    else:
        cell_means = cell_means_raw

    # Build matrix (cells x markers) for cell typing from the THRESHOLDED
    # means, to match the notebook: threshold -> quantify -> arcsinh -> argmax.
    markers_list = list(marker_colors.keys())
    available_markers = [m for m in markers_list if m in cell_means and cell_means[m].size > 0]
    if not available_markers:
        return {}, cell_means

    X = np.column_stack([cell_means[m] for m in available_markers])
    X_trans = np.arcsinh(X)
    argmax_idx = np.argmax(X_trans, axis=1)

    # A cell with all-zero thresholded expression across every marker has no
    # real signal above any threshold; argmax would otherwise arbitrarily
    # pick index 0. Flag these as "Unknown" rather than mistyping them.
    all_zero = np.all(X == 0, axis=1)
    cell_types = [
        "Unknown" if all_zero[i] else ct_marker_dict.get(available_markers[idx], "Unknown")
        for i, idx in enumerate(argmax_idx)
    ]

    predicted_masks = {}
    for marker in available_markers:
        celltype = ct_marker_dict.get(marker, None)
        if celltype is None:
            continue
        mask_cells = np.array([ct == celltype for ct in cell_types])
        cell_indices = np.where(mask_cells)[0]
        label_values = cell_indices + 1
        mask_2d = np.isin(labels, label_values)  # report against full labels
        predicted_masks[marker] = mask_2d

    return predicted_masks, cell_means

def save_thresholds_to_file(thresholds, marker_colors, image_data_keys, output_dir, base_filename):
    """Save current thresholds (fractional or absolute) to a CSV file."""
    markers = sorted([m for m in marker_colors if m in image_data_keys])
    if not markers:
        return False
    out_path = Path(output_dir) / base_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_path, 'w') as f:
            f.write("channel_id,threshold_value\n")
            for marker in markers:
                th = thresholds.get(marker, np.nan)
                f.write(f"{marker},{th:.6f}\n")
        return True
    except Exception:
        return False