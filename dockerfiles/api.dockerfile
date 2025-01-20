# Base image - using python slim image which has ARM support
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /app

# Copy project files
COPY src src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY configs configs/
COPY models models/

# Create data directory structure
RUN mkdir -p /app/data/processed/train

# Install dependencies and project package with verbose output
RUN pip install -r requirements.txt --no-cache-dir -v && \
    pip install -e . --no-deps --no-cache-dir -v

# Set environment variables for non-sensitive data
ENV PYTHONPATH=/app

# Default command to run api
ENTRYPOINT ["python", "-u", "src/ml_ops_project/api.py", "--model_path", "models/model.ckpt"]
