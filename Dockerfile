# Use Python base image with CPU
FROM python:3.10-slim

# Environment setup
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Preinstall Python base tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install base ML + tokenizers + NLP tools
RUN pip install --no-cache-dir \
    transformers>=4.33.2 \
    sentencepiece \
    sacremoses \
    nltk \
    pandas \
    regex \
    mock \
    mosestokenizer \
    bitsandbytes \
    scipy \
    accelerate \
    datasets

# Download NLTK data
RUN python3 -c "import nltk; nltk.download('punkt')"

# Install FastAPI app dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic psutil

# Clone IndicTransToolkit directly to app and install editable
RUN git clone https://github.com/VarunGumma/IndicTransToolkit.git /app/IndicTransToolkit && \
    pip install -e /app/IndicTransToolkit

    # rm -rf /tmp/IndicTransToolkit

# Copy app source
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Health route port
EXPOSE 7860

# Optional healthcheck (requires you to define GET /health)
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run FastAPI
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
