# =============================================================================
# Face Recognition Service — Multi-stage Docker build
# =============================================================================
# Stage 1: Build dependencies (keeps final image smaller)
# Stage 2: Runtime image with only what's needed
# =============================================================================

FROM python:3.11-slim AS builder

WORKDIR /build

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# =============================================================================
# Runtime
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

# Install runtime-only system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create storage directory
RUN mkdir -p /app/storage/faces

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Single Uvicorn process per container (FastAPI best practice for k8s)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
