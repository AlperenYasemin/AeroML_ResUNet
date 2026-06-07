#!/bin/bash
# AeroML Docker Entrypoint
#
# Starts the FastAPI server with static file serving for the frontend.
# Uses PORT env var (set by Cloud Run) or defaults to 8000.

PORT="${PORT:-8000}"

exec python -m uvicorn src.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --log-level info
