"""
Analytical NACA 4-Digit Airfoil SDF (Signed Distance Field) Generator.

Generates 256x256 SDF tensors for NACA 4-digit airfoils on the same spatial
domain used during training: x ∈ [-1, 2], y ∈ [-1, 1].

The SDF encodes geometry: positive values = distance outside the airfoil,
zero/near-zero = on the surface. This matches the AirfRANS convention where
the SDF channel represents the shortest Euclidean distance to the wing surface.
"""

import numpy as np
import torch
from functools import lru_cache


# Spatial domain matching the training preprocessing
X_MIN, X_MAX = -1.0, 2.0
Y_MIN, Y_MAX = -1.0, 1.0
GRID_SIZE = 256


def _naca4_thickness(x: np.ndarray, t: float) -> np.ndarray:
    """
    Compute the half-thickness distribution for a NACA 4-digit airfoil.
    
    Parameters:
        x: Chordwise positions (0 to 1, normalized by chord length).
        t: Maximum thickness as fraction of chord (e.g., 0.12 for NACA 0012).
    
    Returns:
        Half-thickness values at each x position.
    """
    return 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4  # Closed trailing edge coefficient
    )


def _naca4_camber(x: np.ndarray, m: float, p: float):
    """
    Compute the camber line and its gradient for a NACA 4-digit airfoil.
    
    Parameters:
        x: Chordwise positions (0 to 1).
        m: Maximum camber as fraction of chord.
        p: Location of maximum camber as fraction of chord.
    
    Returns:
        yc: Camber line y-coordinates.
        dyc: Camber line gradients (dy/dx).
    """
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    
    if m == 0 or p == 0:
        return yc, dyc
    
    # Forward of max camber
    fwd = x <= p
    yc[fwd] = (m / p**2) * (2.0 * p * x[fwd] - x[fwd]**2)
    dyc[fwd] = (2.0 * m / p**2) * (p - x[fwd])
    
    # Aft of max camber
    aft = x > p
    yc[aft] = (m / (1 - p)**2) * ((1 - 2*p) + 2*p*x[aft] - x[aft]**2)
    dyc[aft] = (2.0 * m / (1 - p)**2) * (p - x[aft])
    
    return yc, dyc


def _generate_naca4_surface_points(naca_code: str, n_points: int = 400) -> np.ndarray:
    """
    Generate the (x, y) surface points of a NACA 4-digit airfoil.
    
    The airfoil is centered at the origin with chord length = 1 (x from 0 to 1).
    Uses cosine spacing for denser point distribution at leading/trailing edges.
    
    Parameters:
        naca_code: 4-digit string, e.g., "0012", "2412", "4415".
        n_points: Number of points along each surface (upper and lower).
    
    Returns:
        Array of shape (2*n_points, 2) with [x, y] surface coordinates.
    """
    # Parse NACA digits
    m = int(naca_code[0]) / 100.0  # Max camber
    p = int(naca_code[1]) / 10.0   # Max camber location
    t = int(naca_code[2:4]) / 100.0  # Max thickness
    
    # Cosine spacing for better leading edge resolution
    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(beta))
    
    yt = _naca4_thickness(x, t)
    yc, dyc = _naca4_camber(x, m, p)
    theta = np.arctan(dyc)
    
    # Upper surface
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    
    # Lower surface
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    # Combine: upper surface (LE to TE) + lower surface (TE to LE)
    upper = np.column_stack([xu, yu])
    lower = np.column_stack([xl, yl])
    
    return np.vstack([upper, lower[::-1]])


def compute_sdf_grid(naca_code: str = "0012") -> np.ndarray:
    """
    Compute a 256x256 Signed Distance Field for a NACA airfoil.
    
    The SDF represents the shortest Euclidean distance from each grid point
    to the airfoil surface. Interior points are clamped to 0 to match the
    convention in the training data (AirfRANS SDF: 0 inside/on surface,
    positive outside with increasing distance).
    
    Parameters:
        naca_code: NACA 4-digit identifier (e.g., "0012", "2412").
    
    Returns:
        SDF grid of shape (256, 256) as a numpy array.
    """
    surface = _generate_naca4_surface_points(naca_code, n_points=500)
    
    # Create the same grid as in training preprocessing
    grid_x, grid_y = np.mgrid[X_MIN:X_MAX:256j, Y_MIN:Y_MAX:256j]
    
    # Flatten grid for vectorized distance computation
    gx_flat = grid_x.ravel()
    gy_flat = grid_y.ravel()
    
    # Compute minimum distance from each grid point to the surface
    # Process in chunks to manage memory
    chunk_size = 1024
    sdf_flat = np.full(len(gx_flat), np.inf)
    
    for start in range(0, len(gx_flat), chunk_size):
        end = min(start + chunk_size, len(gx_flat))
        dx = gx_flat[start:end, None] - surface[:, 0][None, :]
        dy = gy_flat[start:end, None] - surface[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        sdf_flat[start:end] = dist.min(axis=1)
    
    # Detect interior points via ray-casting (point-in-polygon test)
    # Cast horizontal ray from each point to +x∞ and count crossings
    n_surf = len(surface)
    inside = np.zeros(len(gx_flat), dtype=bool)
    
    for start in range(0, len(gx_flat), chunk_size):
        end = min(start + chunk_size, len(gx_flat))
        px = gx_flat[start:end]
        py = gy_flat[start:end]
        crossings = np.zeros(end - start, dtype=int)
        
        for k in range(n_surf):
            x1, y1 = surface[k]
            x2, y2 = surface[(k + 1) % n_surf]
            
            # Does the edge cross the horizontal ray from (px, py) to +∞?
            cond1 = (y1 <= py) & (y2 > py)   # upward crossing
            cond2 = (y2 <= py) & (y1 > py)   # downward crossing
            edge_mask = cond1 | cond2
            
            if not np.any(edge_mask):
                continue
            
            # Compute x-intersection of edge with the horizontal line y = py
            t = (py[edge_mask] - y1) / (y2 - y1)
            x_intersect = x1 + t * (x2 - x1)
            
            # Count if intersection is to the right of the point
            crossings[edge_mask] += (x_intersect > px[edge_mask]).astype(int)
        
        inside[start:end] = (crossings % 2) == 1
    
    # Clamp interior to 0 (matching training data convention)
    sdf_flat[inside] = 0.0
    
    sdf = sdf_flat.reshape(256, 256)
    return sdf.astype(np.float32)


@lru_cache(maxsize=32)
def get_sdf_tensor(naca_code: str = "0012") -> torch.Tensor:
    """
    Get the SDF tensor for a NACA airfoil, with caching.
    
    Returns:
        Tensor of shape (1, 256, 256) — ready to be used as Channel 0 of model input.
    """
    sdf = compute_sdf_grid(naca_code)
    return torch.tensor(sdf, dtype=torch.float32).unsqueeze(0)
