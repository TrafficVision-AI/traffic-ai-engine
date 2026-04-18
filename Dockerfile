FROM python:3.9-slim

# Force native Hugging Face non-root setup
RUN useradd -m -u 1000 user

# Force all Python cache directories to point exclusively to /tmp to avoid "Read-only file system" restrictions natively
ENV MPLCONFIGDIR=/tmp
ENV YOLO_CONFIG_DIR=/tmp
ENV TRANSFORMERS_CACHE=/tmp
ENV XDG_CACHE_HOME=/tmp

WORKDIR /app

# Ensure we have explicit compiled dependencies before attempting python setup installations (cv2 and postgres logic)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    gcc \
    python3-dev \
    libpq-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Establish cache arrays efficiently
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Target complete repository mirroring
COPY --chown=user . /app
RUN chmod -R 777 /tmp

# Explicit runtime profile assignment ensuring strict conformity to platform demands
ENV HOME=/home/user
USER user

# Render explicit porting targeting local environments
EXPOSE 7860
COPY best.pt /app/best.pt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
