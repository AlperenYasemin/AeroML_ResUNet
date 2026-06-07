"""
Model Loading & Inference Pipeline.

Handles cold-start model loading, input tensor construction, and prediction.
The model is loaded once at startup and kept in memory for fast inference.
"""

import os
import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional

from src.model import AeroResUNet
from src.sdf import get_sdf_tensor

# Grid size must match training preprocessing
GRID_SIZE = 256

# Normalization constants (from notebook preprocessing)
# Velocities were normalized by dividing by 100.0
VELOCITY_NORM = 100.0

# Module-level state
_model: Optional[AeroResUNet] = None
_device: Optional[torch.device] = None

# Precomputed SDF cache from real training data
_sample_sdfs: dict[int, torch.Tensor] = {}
# Metadata catalog for each sample (thickness, camber, velocity, aoa)
_sample_catalog: list[dict] = []


def get_device() -> torch.device:
    """Get the current compute device."""
    return _device


def load_model(weights_path: str = None) -> None:
    """
    Load the AeroResUNet model weights into memory.
    
    Called once at server startup (cold start). The model is set to eval()
    mode and moved to the best available device (CUDA > MPS > CPU).
    
    Parameters:
        weights_path: Path to the .pth weights file. Defaults to
                      'aero_resunet_v2_perfect.pth' in the project root.
    """
    global _model, _device
    
    # Device selection
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")
    
    print(f"[AeroML] Device: {_device.type.upper()}")
    
    # Resolve weights path
    if weights_path is None:
        project_root = Path(__file__).parent.parent
        weights_path = str(project_root / "aero_resunet_v2_perfect.pth")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at: {weights_path}\n"
            "Please ensure 'aero_resunet_v2_perfect.pth' is in the project root."
        )
    
    # Load model
    _model = AeroResUNet(in_channels=3, out_channels=3).to(_device)
    _model.load_state_dict(
        torch.load(weights_path, map_location=_device, weights_only=True)
    )
    _model.eval()
    
    param_count = sum(p.numel() for p in _model.parameters())
    print(f"[AeroML] Model loaded: {param_count:,} parameters")
    print(f"[AeroML] Weights: {weights_path}")


def load_sample_sdfs(data_dir: str = None, max_samples: int = 9999) -> None:
    """
    Preload SDF channels from training data.

    Tries compressed cache (sdf_cache.pt) first for fast loading,
    falls back to individual .pt files if cache doesn't exist.
    """
    global _sample_sdfs, _sample_catalog

    project_root = Path(__file__).parent.parent
    cache_path = project_root / "sdf_cache.pt"

    # Try compressed cache first (100MB vs 1.2GB, much faster to load)
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        catalog = []
        for idx, entry in cache.items():
            _sample_sdfs[idx] = entry['sdf'].float()  # back to float32
            catalog.append(entry['metadata'])
        _sample_catalog = sorted(catalog, key=lambda x: x['idx'])
        print(f"[AeroML] Loaded {len(cache)} sample SDFs from cache")
        if _sample_catalog:
            thicknesses = [c['thickness'] for c in _sample_catalog]
            print(f"[AeroML] Thickness range: {min(thicknesses):.1f}% – {max(thicknesses):.1f}%")
            velocities = [c['velocity'] for c in _sample_catalog]
            print(f"[AeroML] Velocity range: {min(velocities):.0f} – {max(velocities):.0f} m/s")
        return

    # Fall back to individual .pt files
    if data_dir is None:
        data_dir = str(project_root / "tensordata" / "train")

    if not os.path.exists(data_dir):
        print(f"[AeroML] Warning: No SDF data found (no cache or training data)")
        print("[AeroML] Falling back to analytical SDF generation only.")
        return

    import math

    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
    count = 0
    catalog = []
    for f in files:
        if count >= max_samples:
            break
        try:
            idx = int(f.replace("sample_", "").replace(".pt", ""))
            data = torch.load(
                os.path.join(data_dir, f),
                map_location="cpu",
                weights_only=False
            )
            inp = data['input']  # [3, 256, 256]
            _sample_sdfs[idx] = inp[0:1]

            sdf_np = inp[0].numpy()
            ux_val = inp[1, 128, 128].item() * 100.0
            uy_val = inp[2, 128, 128].item() * 100.0
            vel = math.sqrt(ux_val**2 + uy_val**2)
            aoa = math.degrees(math.atan2(uy_val, ux_val))

            mid_x = 128
            col = sdf_np[mid_x, :]
            boundary = [j for j in range(256) if abs(col[j]) < 0.02]
            thickness_est = (max(boundary) - min(boundary)) / 256 * 200 if len(boundary) >= 2 else 12.0

            if len(boundary) >= 2:
                center = (max(boundary) + min(boundary)) / 2
                camber_est = abs(center - 128) / 256 * 200
            else:
                camber_est = 0.0

            catalog.append({
                "idx": idx,
                "thickness": round(thickness_est, 1),
                "camber": round(camber_est, 1),
                "velocity": round(vel, 1),
                "aoa": round(aoa, 1),
            })
            count += 1
        except Exception as e:
            print(f"[AeroML] Warning: Could not load {f}: {e}")

    _sample_catalog = catalog
    print(f"[AeroML] Loaded {count} sample SDFs from training data")
    if catalog:
        thicknesses = [c['thickness'] for c in catalog]
        print(f"[AeroML] Thickness range: {min(thicknesses):.1f}% – {max(thicknesses):.1f}%")
        velocities = [c['velocity'] for c in catalog]
        print(f"[AeroML] Velocity range: {min(velocities):.0f} – {max(velocities):.0f} m/s")


def get_available_samples() -> list[int]:
    """Return list of available sample indices with preloaded SDFs."""
    return sorted(_sample_sdfs.keys())


def get_sample_catalog() -> list[dict]:
    """Return metadata catalog for all loaded samples."""
    return _sample_catalog


def build_input_tensor(
    ux_in: float,
    uy_in: float,
    airfoil: str = "naca0012",
    sample_idx: int = None
) -> torch.Tensor:
    """
    Construct the 3-channel input tensor for the model.
    
    Channel 0: SDF (geometry encoding)
    Channel 1: Ux freestream velocity (constant field, normalized)
    Channel 2: Uy freestream velocity (constant field, normalized)
    
    Parameters:
        ux_in: Freestream velocity in X direction (m/s, raw units).
        uy_in: Freestream velocity in Y direction (m/s, raw units).
        airfoil: NACA 4-digit code (e.g., "naca0012"). Used when sample_idx is None.
        sample_idx: If provided, use the preloaded SDF from this training sample.
    
    Returns:
        Tensor of shape [1, 3, 256, 256] ready for model inference.
    """
    # Get SDF channel
    if sample_idx is not None and sample_idx in _sample_sdfs:
        sdf = _sample_sdfs[sample_idx]
    else:
        # Use analytical SDF - strip "naca" prefix if present
        naca_code = airfoil.lower().replace("naca", "").strip()
        if len(naca_code) != 4 or not naca_code.isdigit():
            naca_code = "0012"  # Default to NACA 0012
        sdf = get_sdf_tensor(naca_code)
    
    # Build velocity channels (normalized, matching training preprocessing)
    ux_normalized = ux_in / VELOCITY_NORM
    uy_normalized = uy_in / VELOCITY_NORM
    
    ux_channel = torch.full((1, GRID_SIZE, GRID_SIZE), ux_normalized, dtype=torch.float32)
    uy_channel = torch.full((1, GRID_SIZE, GRID_SIZE), uy_normalized, dtype=torch.float32)
    
    # Stack: [3, 256, 256]
    input_tensor = torch.cat([sdf, ux_channel, uy_channel], dim=0)
    
    # Add batch dimension: [1, 3, 256, 256]
    return input_tensor.unsqueeze(0)


def predict(
    ux_in: float,
    uy_in: float,
    airfoil: str = "naca0012",
    sample_idx: int = None
) -> dict:
    """
    Run model inference for given boundary conditions.
    
    Parameters:
        ux_in: Freestream velocity in X direction (m/s).
        uy_in: Freestream velocity in Y direction (m/s).
        airfoil: NACA 4-digit code (e.g., "naca0012").
        sample_idx: Optional sample index for preloaded SDF.
    
    Returns:
        Dictionary with:
            - "prediction": Raw output tensor [1, 3, 256, 256]
            - "inference_time_ms": Inference time in milliseconds
            - "device": Device used for computation
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    # Build input tensor and move to device
    input_tensor = build_input_tensor(ux_in, uy_in, airfoil, sample_idx)
    input_tensor = input_tensor.to(_device)
    
    # Run inference
    start = time.perf_counter()
    with torch.no_grad():
        output = _model(input_tensor)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    
    # Extract SDF from input for client-side airfoil masking
    sdf_channel = input_tensor[0, 0].cpu()  # [256, 256]
    
    return {
        "prediction": output.cpu(),
        "sdf": sdf_channel,
        "inference_time_ms": round(elapsed_ms, 2),
        "device": _device.type,
    }
