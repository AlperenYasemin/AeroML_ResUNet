"""
AeroML — FastAPI Backend Server.

Production-grade REST API wrapping the AeroResUNet inference engine.
Accepts boundary conditions (freestream velocities), runs the deep learning
model, renders colored heatmaps server-side, and returns Base64-encoded
PNG images in a JSON response.

Usage:
    uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import time
import base64
import numpy as np
import torch
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.inference import load_model, load_sample_sdfs, predict, get_available_samples, get_sample_catalog, get_device
from src.renderer import render_prediction


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for the /predict endpoint."""
    ux_in: float = Field(
        ...,
        description="Freestream velocity in X direction (m/s). Typical range: 10–100.",
        examples=[46.0],
    )
    uy_in: float = Field(
        ...,
        description="Freestream velocity in Y direction (m/s). Encodes angle of attack.",
        examples=[5.0],
    )
    airfoil: str = Field(
        default="naca0012",
        description="NACA 4-digit airfoil code (e.g., 'naca0012', 'naca2412').",
        examples=["naca0012"],
    )
    sample_idx: int | None = Field(
        default=None,
        description="Optional: use a preloaded SDF from training data instead of analytical generation.",
        examples=[0],
    )


class PredictResponse(BaseModel):
    """Response body from the /predict endpoint."""
    pressure: str = Field(description="Base64-encoded PNG heatmap of the pressure field.")
    velocity_x: str = Field(description="Base64-encoded PNG heatmap of the Ux velocity field.")
    velocity_y: str = Field(description="Base64-encoded PNG heatmap of the Uy velocity field.")
    inference_time_ms: float = Field(description="Model inference time in milliseconds.")
    device: str = Field(description="Compute device used (cpu, cuda, mps).")


class RawPredictResponse(BaseModel):
    """Response body from the /predict/raw endpoint. Fields are base64-encoded float32 arrays."""
    width: int = Field(description="Grid width (256).")
    height: int = Field(description="Grid height (256).")
    pressure: str = Field(description="Base64-encoded float32 array (256*256 values).")
    velocity_x: str = Field(description="Base64-encoded float32 array.")
    velocity_y: str = Field(description="Base64-encoded float32 array.")
    sdf: str = Field(description="Base64-encoded float32 SDF array for airfoil masking.")
    pressure_min: float
    pressure_max: float
    velocity_x_min: float
    velocity_x_max: float
    velocity_y_min: float
    velocity_y_max: float
    inference_time_ms: float
    device: str


class HealthResponse(BaseModel):
    """Response body for health check endpoints."""
    status: str
    device: str
    model_loaded: bool
    available_samples: list[int]


class SampleCatalogResponse(BaseModel):
    """Response body for the sample catalog endpoint."""
    total: int
    samples: list[dict]


# ---------------------------------------------------------------------------
# Application Lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    On startup (cold start):
    - Load model weights into RAM/VRAM
    - Preload sample SDFs from training data
    
    This eliminates per-request I/O latency.
    """
    print("=" * 60)
    print("  AeroML Inference Server — Starting Up")
    print("=" * 60)
    
    start = time.time()
    load_model()
    load_sample_sdfs()  # Load ALL training samples with metadata
    elapsed = time.time() - start
    
    print(f"[AeroML] Cold start completed in {elapsed:.2f}s")
    print("=" * 60)
    
    yield  # Server is running
    
    # Shutdown
    print("[AeroML] Server shutting down.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AeroML Inference API",
    description=(
        "Data-driven surrogate model for 2D airfoil aerodynamics. "
        "Predicts pressure and velocity fields around NACA airfoils "
        "in ~40ms using a Deep Residual U-Net (AeroResUNet) trained "
        "on the AirfRANS dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for frontend dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status", response_model=HealthResponse, tags=["Health"])
async def api_status():
    """Health check — returns server status and device info."""
    device = get_device()
    return HealthResponse(
        status="online",
        device=device.type if device else "not loaded",
        model_loaded=device is not None,
        available_samples=get_available_samples(),
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring systems."""
    device = get_device()
    return HealthResponse(
        status="online",
        device=device.type if device else "not loaded",
        model_loaded=device is not None,
        available_samples=get_available_samples(),
    )


@app.get("/samples/catalog", response_model=SampleCatalogResponse, tags=["Data"])
async def samples_catalog():
    """
    Return metadata catalog for all loaded training samples.
    
    Each sample includes estimated geometric properties (thickness, camber)
    and the original training flow conditions (velocity, angle of attack).
    """
    catalog = get_sample_catalog()
    return SampleCatalogResponse(
        total=len(catalog),
        samples=catalog,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_endpoint(request: PredictRequest):
    """
    Run aerodynamic field prediction (server-rendered PNG heatmaps).
    
    The system is stateless — no prediction history is stored.
    """
    try:
        result = predict(
            ux_in=request.ux_in,
            uy_in=request.uy_in,
            airfoil=request.airfoil,
            sample_idx=request.sample_idx,
        )
        
        heatmaps = render_prediction(result["prediction"])
        
        return PredictResponse(
            pressure=heatmaps["pressure"],
            velocity_x=heatmaps["velocity_x"],
            velocity_y=heatmaps["velocity_y"],
            inference_time_ms=result["inference_time_ms"],
            device=result["device"],
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Render inference failed: {str(e)}")


@app.post("/predict/raw", response_model=RawPredictResponse, tags=["Inference"])
async def predict_raw_endpoint(request: PredictRequest):
    """
    Run aerodynamic field prediction (raw tensor data).
    
    Returns base64-encoded float32 arrays for client-side rendering.
    This enables interactive Canvas-based visualization with smooth
    animations and particle flow overlays.
    """
    try:
        result = predict(
            ux_in=request.ux_in,
            uy_in=request.uy_in,
            airfoil=request.airfoil,
            sample_idx=request.sample_idx,
        )
        
        pred = result["prediction"].squeeze(0).numpy()  # [3, 256, 256]
        sdf = result["sdf"].numpy()  # [256, 256]
        
        p_field = pred[0].astype(np.float32)
        ux_field = pred[1].astype(np.float32)
        uy_field = pred[2].astype(np.float32)
        sdf_field = sdf.astype(np.float32)
        
        return RawPredictResponse(
            width=256,
            height=256,
            pressure=base64.b64encode(p_field.tobytes()).decode(),
            velocity_x=base64.b64encode(ux_field.tobytes()).decode(),
            velocity_y=base64.b64encode(uy_field.tobytes()).decode(),
            sdf=base64.b64encode(sdf_field.tobytes()).decode(),
            pressure_min=float(p_field.min()),
            pressure_max=float(p_field.max()),
            velocity_x_min=float(ux_field.min()),
            velocity_x_max=float(ux_field.max()),
            velocity_y_min=float(uy_field.min()),
            velocity_y_max=float(uy_field.max()),
            inference_time_ms=result["inference_time_ms"],
            device=result["device"],
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


# ---------------------------------------------------------------------------
# Static Frontend Serving (Docker production mode)
# ---------------------------------------------------------------------------
# When the built frontend exists at ./static (inside Docker), serve it.
# In development this directory doesn't exist, so nothing is mounted.

_static_dir = Path(__file__).parent.parent / "static"
if _static_dir.is_dir():
    @app.get("/app/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve the React SPA — return index.html for all non-asset routes."""
        file_path = _static_dir / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(_static_dir / "index.html")

    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")

