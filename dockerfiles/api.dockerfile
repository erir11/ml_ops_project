# Use a specific Python version
FROM python:3.11-slim AS base

# Install required system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements files first for better caching
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt

# Install dependencies from requirements files
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir --verbose

# Copy the rest of the application files
COPY src/ src/
COPY README.md README.md
COPY pyproject.toml pyproject.toml

# Install the package in editable mode
RUN pip install . --no-deps --no-cache-dir --verbose

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Define the entry point for the container
ENTRYPOINT ["uvicorn", "src.ml_ops_project.api:app", "--host", "0.0.0.0", "--port", "8000"]
