# ─────────────────────────────────────────────
# FinDocAgent-Env — Dockerfile
# Compatible with Hugging Face Spaces (port 7860)
# ─────────────────────────────────────────────

FROM python:3.10-slim

# Metadata
LABEL maintainer="your-hf-username"
LABEL description="FinDocAgent-Env — OpenEnv environment for financial document processing"
LABEL version="1.0.0"

# System deps
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . ./findocagent_env/

# Set Python path so imports resolve correctly
ENV PYTHONPATH=/app/findocagent_env

# HF Spaces requires port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start server
CMD ["uvicorn", "findocagent_env.server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "2"]