FROM python:3.12-slim

# Render's native (buildpack) Python runtime cannot run apt-get at all -- its build container's
# /var/lib/apt/lists is read-only, confirmed live (see PR discussion). Docker is the only way to
# get tesseract (OCR fallback, syspare_rag/config.py's ocr_lang) and the other system libraries
# this dependency set needs onto the actual runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-mya \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

COPY . .

ENV PATH="/app/.venv/bin:$PATH"

# $PORT is injected by Render at container start -- shell form (not exec-array form) so it
# actually gets substituted.
CMD gunicorn rag_server:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
