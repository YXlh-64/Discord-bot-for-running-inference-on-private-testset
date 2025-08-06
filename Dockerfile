FROM python:3.10-slim

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create a non-root user
RUN useradd -m runner
USER runner

WORKDIR /workspace

# Entrypoint will be set dynamically