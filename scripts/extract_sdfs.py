"""
Extract SDF channels from training data into a single lightweight file.

This creates a compact sdf_cache.pt (~15MB) instead of needing the full
1.2GB tensordata/train/ directory at runtime. Used for Docker/cloud deployment.

Usage:
    python scripts/extract_sdfs.py
"""

import os
import sys
import torch
import math
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent
    train_dir = project_root / "tensordata" / "train"
    output_path = project_root / "sdf_cache.pt"

    if not train_dir.exists():
        print(f"Error: {train_dir} not found")
        sys.exit(1)

    files = sorted([f for f in os.listdir(train_dir) if f.endswith('.pt')])
    print(f"Processing {len(files)} samples from {train_dir}")

    cache = {}
    for f in files:
        try:
            idx = int(f.replace("sample_", "").replace(".pt", ""))
            data = torch.load(train_dir / f, map_location="cpu", weights_only=False)
            inp = data['input']  # [3, 256, 256]

            # Extract SDF (channel 0) and metadata
            sdf = inp[0:1]  # [1, 256, 256]
            ux_val = inp[1, 128, 128].item() * 100.0
            uy_val = inp[2, 128, 128].item() * 100.0
            vel = math.sqrt(ux_val**2 + uy_val**2)
            aoa = math.degrees(math.atan2(uy_val, ux_val))

            # Estimate thickness from SDF
            sdf_np = inp[0].numpy()
            mid_x = 128
            col = sdf_np[mid_x, :]
            boundary = [j for j in range(256) if abs(col[j]) < 0.02]
            thickness_est = (max(boundary) - min(boundary)) / 256 * 200 if len(boundary) >= 2 else 12.0

            if len(boundary) >= 2:
                center = (max(boundary) + min(boundary)) / 2
                camber_est = abs(center - 128) / 256 * 200
            else:
                camber_est = 0.0

            cache[idx] = {
                'sdf': sdf,
                'thickness': round(thickness_est, 1),
                'camber': round(camber_est, 1),
                'velocity': round(vel, 1),
                'aoa': round(aoa, 1),
            }
        except Exception as e:
            print(f"  Warning: skipping {f}: {e}")

    torch.save(cache, output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved {len(cache)} SDFs to {output_path} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
