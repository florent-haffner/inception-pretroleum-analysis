# syntax=docker/dockerfile:1
# Build from the repository root:  docker build -f env/Dockerfile -t inception-petroleum-analysis .
FROM python:3.13-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MPLCONFIGDIR=/tmp/matplotlib \
    KERAS_HOME=/tmp/.keras

WORKDIR /app

# Install dependencies first to benefit from layer caching
COPY env/requirements.txt env/requirements.txt
RUN pip install -r env/requirements.txt

# Run as an unprivileged user
RUN useradd --create-home --uid 1000 app && mkdir -p /app/model && chmod 777 /app/model

# Only copy what the Python pipeline needs
COPY --chown=app:app src/ src/
COPY --chown=app:app scripts/ scripts/
COPY --chown=app:app data/ data/

USER app

# Trained weights are written here; mount a volume to keep them
VOLUME ["/app/model"]

CMD ["python", "-m", "scripts.models-calibration"]
