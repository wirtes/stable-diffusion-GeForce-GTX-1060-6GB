# Multi-stage Dockerfile for Stable Diffusion API with CUDA support

# Stage 1: Base image with CUDA runtime
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as base

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-pip \
    python3.9-dev \
    python3.9-venv \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symbolic link for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip

# Stage 2: Dependencies installation
FROM base as dependencies

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with CUDA support
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt

# Stage 3: Application
FROM dependencies as application

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/cache

# Set proper permissions
RUN chmod +x /app/main.py

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check with proper GPU validation
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Startup command with proper GPU initialization
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]