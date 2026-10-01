# ==============================================================================
# AirIntel Production Container Definition
# Author: Ritvika (IIT BHU)
# Base Image: Official Python 3.12 Slim Linux (Debian Bookworm)
# ==============================================================================
FROM python:3.12-slim AS base

# Environment settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

# Install lightweight system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first for optimal Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and assets
COPY app.py .
COPY pytest.ini .
COPY dvc.yaml .
COPY assets/ assets/
COPY components/ components/
COPY models/deployment/ models/deployment/
COPY reports/ reports/
COPY src/ src/
COPY tests/ tests/
COPY data/processed/clean_airintel.parquet data/processed/clean_airintel.parquet
RUN mkdir -p logs

# Create non-root system user for security compliance
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENV PORT=8501
EXPOSE 8501 8080

ENTRYPOINT ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
