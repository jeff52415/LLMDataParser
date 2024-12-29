# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# Set working directory
WORKDIR /app

# Create cache directories for Hugging Face and set permissions correctly
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface && \
    mkdir -p /app/.cache/torch && \
    mkdir -p /app/.cache/transformers

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pip \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install poetry
RUN pip install poetry==${POETRY_VERSION} && \
    poetry config virtualenvs.create false

# Copy dependency files first
COPY pyproject.toml poetry.lock ./

# Install dependencies using the lock file
RUN poetry install --no-dev --no-interaction --no-ansi

# Create app user and group with specific UID/GID
RUN groupadd -r app --gid 1000 && \
    useradd -r -g app --uid 1000 --create-home app

# Set ownership of all cache directories
RUN chown -R app:app /app/.cache && \
    chmod -R 755 /app/.cache

# Before switching to non-root user, create and set permissions
RUN mkdir -p /home/app/.cache && \
    mkdir -p /home/app/.config/matplotlib && \
    chown -R app:app /home/app/.cache && \
    chown -R app:app /home/app/.config

# Set matplotlib config dir
ENV MPLCONFIGDIR=/home/app/.config/matplotlib

# Switch to non-root user
USER app

# Copy the rest of the application
COPY --chown=app:app . .

# Expose the port the app runs on
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
