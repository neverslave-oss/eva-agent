# ======= BUILDER STAGE =======
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies and audio/media libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    libsndfile1 \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124
RUN pip install --user --no-cache-dir -r requirements.txt

# ======= RUNTIME STAGE =======
FROM python:3.11-slim

# Install only runtime audio/media libraries (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Ensure pip installs are in PATH
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Copy application code
COPY src/ ./src/
COPY config.container.yaml config.yaml
COPY kernel .
RUN chmod +x kernel
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create runtime directories
RUN mkdir -p /app/skills /app/routines /app/workspace

# Model directories will be mounted from host
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8779/health || exit 1

EXPOSE 8779
WORKDIR /app/src

CMD ["/app/entrypoint.sh"]
