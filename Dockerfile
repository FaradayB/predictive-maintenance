FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the package so `predictivecare` is importable (deps already installed)
RUN pip install --no-cache-dir -e . --no-deps

# Train the classifiers into models/ (dataset is committed; models are gitignored)
RUN python ml/train.py

# Create runtime directories
RUN mkdir -p logs reports chroma_db

# Expose ports
# 8010 = FastAPI, 8501 = Technician, 8502 = Owner, 8000 = Metrics
EXPOSE 8000 8010 8501 8502
