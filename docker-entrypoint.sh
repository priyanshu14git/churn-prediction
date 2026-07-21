#!/bin/bash
# docker-entrypoint.sh
# Starts the FastAPI backend in the background, then the Streamlit frontend
# in the foreground (so the container stays alive as long as Streamlit runs).

set -e

echo "Starting FastAPI backend on port 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Give the API a moment to start before Streamlit tries to call it
sleep 3

echo "Starting Streamlit frontend on port 8501..."
streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501