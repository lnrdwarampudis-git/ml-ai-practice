# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/          ./app/
COPY scripts/      ./scripts/
COPY model_artifacts/ ./model_artifacts/

# Ownership
RUN chown -R appuser:appuser /app
USER appuser

# ── Environment defaults (override at runtime)
ENV MODEL_ARTIFACTS_DIR=/app/model_artifacts \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    PORT=8000 \
    WORKERS=2

EXPOSE 8000

# Healthcheck — Docker will restart if /health returns non-200
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Gunicorn + Uvicorn workers for production concurrency
CMD ["sh", "-c", \
     "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS} --log-config /dev/null"]
