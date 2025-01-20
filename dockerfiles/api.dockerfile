# Use Python 3.11 slim as the base image
FROM python:3.11-slim AS base

# Set the working directory inside the container
WORKDIR /ml_ops_project

# Install system dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src /ml_ops_project/src

# Copy the model explicitly
COPY models /ml_ops_project/models

# Expose the port that FastAPI will run on
EXPOSE 8000

# Set the entry point to start the application
CMD ["uvicorn", "src.ml_ops_project.main:app", "--host", "0.0.0.0", "--port", "8000"]
