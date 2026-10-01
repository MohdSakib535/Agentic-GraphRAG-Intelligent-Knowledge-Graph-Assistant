# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt \
    # Warm the tokenizer cache when the network allows it; the app falls back to an approximation otherwise.
    && (python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" || echo "tiktoken cache not warmed")

COPY . .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/uploads /opt/tiktoken \
    && chown -R appuser:appuser /data /app /opt/tiktoken \
    && chmod +x /app/docker/entrypoint.sh

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/v1/health', timeout=4).status == 200 else 1)"

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["api"]
