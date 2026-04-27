# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for ONNX Runtime + audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into a prefix so we can copy just the site-packages
RUN pip install --upgrade pip \
 && pip install --prefix=/install -r requirements.txt \
 && pip install --prefix=/install redis hiredis

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Outputs directory (MP3s land here; mounted as a PVC in k8s)
RUN mkdir -p outputs

# Non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
RUN chown -R appuser:appgroup /app
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Warm up the ONNX classifier at container start (preload.py already does this)
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Redis config (overridden via k8s env/configmap)
ENV REDIS_HOST=redis-service
ENV REDIS_PORT=6379
ENV REDIS_TTL=3600

CMD ["python", "app.py"]