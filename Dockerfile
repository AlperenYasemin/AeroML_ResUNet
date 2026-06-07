# =============================================================================
# AeroML — Multi-stage Docker Build
#
# Stage 1: Build the React frontend into static files
# Stage 2: Production Python image serving both the API and the frontend
#
# Usage:
#   docker build -t aeroml .
#   docker run -p 8000:8000 aeroml                    # CPU-only
#   docker run --gpus all -p 8000:8000 aeroml          # With NVIDIA GPU
#
# The web frontend is served at http://localhost:8000
# The API is available at    http://localhost:8000/predict/raw
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — Build frontend
# ---------------------------------------------------------------------------
FROM node:20-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund 2>/dev/null || npm install --no-audit --no-fund
COPY frontend/ ./
# Use empty string so API calls go to same origin in production
ENV VITE_API_URL=""
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2 — Production backend + static frontend
# ---------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies — install PyTorch with CUDA support
# The CUDA libraries add ~2GB but enable GPU inference on Cloud Run
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    numpy \
    scipy \
    matplotlib \
    tqdm

# Copy application code
COPY src/ ./src/
COPY aero_resunet_v2_perfect.pth .

# Copy compressed SDF cache (100MB vs 1.2GB full training data)
COPY sdf_cache.pt .

# Copy built frontend static files
COPY --from=frontend-build /app/frontend/dist ./static

# Serve frontend static files from FastAPI
# This is handled by the startup script below

EXPOSE 8000

# Start server — the entrypoint script adds static file serving
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

CMD ["./docker-entrypoint.sh"]
