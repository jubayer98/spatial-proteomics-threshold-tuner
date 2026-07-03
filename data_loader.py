import xarray as xr
import numpy as np

def find_image_data(ds, marker_colors):
    """
    Locate the image DataArray with channel and spatial dimensions.
    Returns:
        image_data: dict mapping marker name -> 2D numpy array (for markers in marker_colors)
        channel_dim: name of the channel dimension
        channel_names: list of all channel names in the dataset
    """
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
        # fallback: any 3‑dim variable with one large dimension
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
        raise ValueError("Could not find image data with a channel dimension and spatial dimensions.")

    # Get channel names
    if channel_dim in image_var.coords:
        channel_names = [str(c) for c in image_var.coords[channel_dim].values]
    else:
        channel_names = [str(i) for i in range(image_var.sizes[channel_dim])]

    # Load raw images for markers in marker_colors
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
        except Exception:
            pass

    return image_data, channel_dim, channel_names

def find_segmentation_labels(ds):
    """Locate a 2D integer label array (segmentation mask)."""
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
    return label_var.values if label_var is not None else None