# Dockerfile
# Runs both the FastAPI backend and Streamlit frontend in one container.

FROM python:3.11-slim

# System dependency needed by lightgbm (Linux equivalent of libomp on Mac)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching - only re-installs
# if requirements.txt changes, not on every code change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose both ports: 8000 for FastAPI, 8501 for Streamlit
EXPOSE 8000 8501

# Start both processes. In production you'd usually split these into two
# separate containers/services, but for a single-container demo deployment
# this is the simplest approach.
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

CMD ["/app/docker-entrypoint.sh"]