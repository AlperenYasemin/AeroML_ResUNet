"""
Heatmap Rendering Pipeline.

Converts raw model output tensors into colored heatmap images,
then encodes them as Base64 PNG strings for JSON transport.

This server-side rendering approach offloads all visualization
compute from client devices (especially weak mobile devices).
"""

import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Spatial domain matching training data
X_MIN, X_MAX = -1.0, 2.0
Y_MIN, Y_MAX = -1.0, 1.0

# Channel names and colormaps
FIELD_CONFIG = {
    0: {"name": "pressure", "label": "Pressure (p)", "cmap": "RdBu_r"},
    1: {"name": "velocity_x", "label": "Velocity Ux", "cmap": "coolwarm"},
    2: {"name": "velocity_y", "label": "Velocity Uy", "cmap": "coolwarm"},
}


def tensor_to_base64_png(
    field: np.ndarray,
    title: str = "",
    cmap: str = "jet",
    figsize: tuple = (8, 4),
    dpi: int = 100,
) -> str:
    """
    Render a 2D field array as a colored heatmap and return as Base64 PNG.
    
    Parameters:
        field: 2D numpy array of shape (256, 256).
        title: Title text for the plot.
        cmap: Matplotlib colormap name.
        figsize: Figure size in inches.
        dpi: Dots per inch for the output image.
    
    Returns:
        Base64-encoded PNG string with data URI prefix.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    
    im = ax.imshow(
        field.T,  # Transpose to match physical orientation
        extent=(X_MIN, X_MAX, Y_MIN, Y_MAX),
        origin='lower',
        cmap=cmap,
        aspect='auto',
    )
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.set_xlabel("x / chord")
    ax.set_ylabel("y / chord")
    fig.colorbar(im, ax=ax, shrink=0.8)
    
    fig.tight_layout()
    
    # Render to in-memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def render_prediction(prediction_tensor, include_titles: bool = True) -> dict:
    """
    Render all three output channels as Base64 heatmap images.
    
    Parameters:
        prediction_tensor: Model output tensor of shape [1, 3, 256, 256].
        include_titles: Whether to add titles to the plots.
    
    Returns:
        Dictionary with keys: "pressure", "velocity_x", "velocity_y",
        each containing a Base64-encoded PNG string.
    """
    pred = prediction_tensor.squeeze(0).numpy()  # [3, 256, 256]
    
    result = {}
    for ch_idx, config in FIELD_CONFIG.items():
        field = pred[ch_idx]  # [256, 256]
        title = config["label"] if include_titles else ""
        
        b64_img = tensor_to_base64_png(
            field=field,
            title=title,
            cmap=config["cmap"],
        )
        result[config["name"]] = b64_img
    
    return result


def render_comparison(prediction_tensor, target_tensor) -> dict:
    """
    Render side-by-side comparison of prediction vs ground truth.
    Useful for validation and QA testing.
    
    Parameters:
        prediction_tensor: Model output [1, 3, 256, 256].
        target_tensor: Ground truth [3, 256, 256].
    
    Returns:
        Dictionary with Base64 comparison images for each field.
    """
    pred = prediction_tensor.squeeze(0).numpy()
    target = target_tensor.numpy()
    
    result = {}
    for ch_idx, config in FIELD_CONFIG.items():
        pred_field = pred[ch_idx]
        target_field = target[ch_idx]
        
        # Shared color scale
        vmin = min(pred_field.min(), target_field.min())
        vmax = max(pred_field.max(), target_field.max())
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        fig, axs = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
        
        im1 = axs[0].imshow(
            target_field.T, extent=(X_MIN, X_MAX, Y_MIN, Y_MAX),
            origin='lower', cmap=config["cmap"], norm=norm, aspect='auto'
        )
        axs[0].set_title("Ground Truth (CFD)", fontsize=11, fontweight='bold')
        axs[0].set_xlabel("x / chord")
        axs[0].set_ylabel("y / chord")
        
        im2 = axs[1].imshow(
            pred_field.T, extent=(X_MIN, X_MAX, Y_MIN, Y_MAX),
            origin='lower', cmap=config["cmap"], norm=norm, aspect='auto'
        )
        axs[1].set_title("AeroML Prediction", fontsize=11, fontweight='bold')
        axs[1].set_xlabel("x / chord")
        axs[1].set_ylabel("y / chord")
        
        fig.colorbar(im2, ax=axs, shrink=0.8, label=config["label"])
        fig.suptitle(config["label"], fontsize=13, fontweight='bold')
        fig.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#1a1a2e')
        plt.close(fig)
        
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        result[config["name"]] = f"data:image/png;base64,{b64}"
    
    return result
