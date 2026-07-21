 #!/bin/bash
# docker-entrypoint.sh
# Starts the FastAPI backend in the background, then the Streamlit frontend
# in the foreground (so the container stays alive as long as Streamlit runs).
#
# On Render (and most cloud hosts), only one public port is exposed, passed
# via the PORT environment variable - so Streamlit (the user-facing app)
# binds to that, while FastAPI stays internal-only on 8000.

set -e

STREAMLIT_PORT="${PORT:-8501}"

echo "Starting FastAPI backend on port 8000 (internal)..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Give the API a moment to start before Streamlit tries to call it
sleep 3

echo "Starting Streamlit frontend on port ${STREAMLIT_PORT}..."
streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port "${STREAMLIT_PORT}"