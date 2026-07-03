"""
Artefact / tissue-exclusion masking, ported from the notebook's
`rasterize_polygon` + `load_zarr` logic so it can be reused in the Streamlit
app.

Usage in app.py (see integration notes at the bottom of this file):

    from artefacts import load_artefact_mask

    cell_mask = load_artefact_mask(
        artefact_csv_path,
        sample_id,
        shape=(ds.sizes['y'], ds.sizes['x'])
    )
    predicted_masks, cell_means = recompute_predictions(
        marker_colors, ct_marker_dict, labels, image_data,
        st.session_state.thresholds, cell_mask=cell_mask
    )
"""
import ast
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from skimage.draw import polygon


def rasterize_polygon(polygon_coords, shape=(2000, 2000)):
    """
    Rasterize an exclusion polygon into a binary NumPy array (y-axis flipped
    to match image coordinates, same convention as the notebook).

    Returns
    -------
    np.ndarray (uint8)
        1 = keep (outside polygon), 0 = excluded (inside polygon).
    """
    poly = Polygon(polygon_coords)
    if not poly.is_valid:
        raise ValueError("The provided polygon is invalid.")

    x, y = zip(*polygon_coords)
    y_flipped = [shape[0] - yi for yi in y]

    rr, cc = polygon(y_flipped, x, shape)

    binary_array = np.ones(shape, dtype=np.uint8)
    binary_array[rr, cc] = 0
    return binary_array


def build_artefact_mask(artefact_df, sample_id, shape):
    """
    Combine all artefact polygons for one sample into a single keep/exclude
    mask, same shape as the image (True = keep, False = excluded).

    Parameters
    ----------
    artefact_df : pd.DataFrame
        Must have columns 'sample' and 'polygon' (polygon stored as a
        string, e.g. "[(x1,y1), (x2,y2), ...]").
    sample_id : str
    shape : tuple (height, width)
    """
    df = artefact_df[artefact_df['sample'] == sample_id]
    if df.shape[0] == 0:
        return np.ones(shape, dtype=bool)

    full_mask = np.ones(shape, dtype=np.uint8)
    for _, row in df.iterrows():
        polygon_coords = ast.literal_eval(row['polygon'])
        mask = rasterize_polygon(polygon_coords, shape=shape)
        full_mask = np.minimum(full_mask, mask)

    return full_mask.astype(bool)


def load_artefact_mask(artefact_csv_path, sample_id, shape):
    """
    Convenience wrapper: load the artefact CSV from disk and build the mask
    for one sample. Returns an all-True mask (nothing excluded) if the CSV
    is missing or unreadable, so the app degrades gracefully rather than
    crashing when no artefact file is configured.
    """
    try:
        artefact_df = pd.read_csv(artefact_csv_path)
    except Exception:
        return np.ones(shape, dtype=bool)

    return build_artefact_mask(artefact_df, sample_id, shape)