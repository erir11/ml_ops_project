# Base image
FROM python:3.11-slim

# Create non-root user
RUN useradd -m appuser

# Install system dependencies and setup directories in single layer
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /app/data/processed/train && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy project files
COPY --chown=appuser:appuser src src/
COPY --chown=appuser:appuser requirements.txt requirements.txt
COPY --chown=appuser:appuser pyproject.toml pyproject.toml
COPY --chown=appuser:appuser models models/

# Switch to non-root user
USER appuser

# Install dependencies
RUN pip install --no-cache-dir -v -r requirements.txt && \
    pip install --no-cache-dir -v -e . --no-deps

# Set environment variables
ENV PYTHONPATH=/app

ENV PORT=8080

# Default command
ENTRYPOINT ["python", "-u", "src/ml_ops_project/api.py", "--model_path", "models/model.ckpt"]
