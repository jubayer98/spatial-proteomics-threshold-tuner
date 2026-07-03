import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def set_black_background(fig, ax):
    """Set figure and axes background to black."""
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

def plot_raw_image(raw_img, color_hex, title="Raw", boundaries=None, figsize=(5,5)):
    """Display raw channel with a black‑to‑color colormap."""
    fig, ax = plt.subplots(figsize=figsize)
    set_black_background(fig, ax)
    p1, p99 = np.percentile(raw_img, (1, 99))
    norm_img = np.clip((raw_img - p1) / (p99 - p1 + 1e-8), 0, 1)
    cmap = LinearSegmentedColormap.from_list('custom', ['black', color_hex])
    ax.imshow(norm_img, cmap=cmap)
    if boundaries is not None:
        ax.imshow(boundaries, cmap='gray', alpha=0.5, interpolation='none')
    ax.set_title(title, color='white', fontsize=12)
    ax.axis('off')
    return fig

def plot_thresholded_intensity(raw_img, threshold, color_hex, title="Thresholded", boundaries=None, figsize=(5,5)):
    """
    Display the thresholded image (values below threshold set to 0)
    using the same black‑to‑color colormap and normalisation as the raw image.
    """
    fig, ax = plt.subplots(figsize=figsize)
    set_black_background(fig, ax)

    # Create thresholded copy
    thresh_img = raw_img.copy()
    thresh_img[thresh_img < threshold] = 0

    # Normalise using percentiles from the raw image (for consistency)
    p1, p99 = np.percentile(raw_img, (1, 99))
    norm_img = np.clip((thresh_img - p1) / (p99 - p1 + 1e-8), 0, 1)

    # Black‑to‑color colormap
    cmap = LinearSegmentedColormap.from_list('custom', ['black', color_hex])
    ax.imshow(norm_img, cmap=cmap)

    if boundaries is not None:
        ax.imshow(boundaries, cmap='gray', alpha=0.5, interpolation='none')

    ax.set_title(title, color='white', fontsize=12)
    ax.axis('off')
    return fig

def plot_predicted_image(predicted_mask, color_hex, title="Predicted", boundaries=None, figsize=(5,5)):
    """Display predicted cell mask in the marker colour on black background."""
    fig, ax = plt.subplots(figsize=figsize)
    set_black_background(fig, ax)
    rgb = np.zeros((*predicted_mask.shape, 3), dtype=np.uint8)
    hex_color = color_hex.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    rgb[predicted_mask] = [r, g, b]
    ax.imshow(rgb)
    if boundaries is not None:
        ax.imshow(boundaries, cmap='gray', alpha=0.5, interpolation='none')
    ax.set_title(title, color='white', fontsize=12)
    ax.axis('off')
    return fig

def plot_composite_image(composite_array, title="Composite", figsize=(4,4)):
    """Display an RGB composite image."""
    fig, ax = plt.subplots(figsize=figsize)
    set_black_background(fig, ax)
    ax.imshow(composite_array)
    ax.set_title(title, color='white', fontsize=10)
    ax.axis('off')
    return fig