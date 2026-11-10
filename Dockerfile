# syntax=docker/dockerfile:1
# api-gateway image. Base images are pinned by digest; never `latest`.

ARG PYTHON_IMAGE=python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168

FROM ${UV_IMAGE} AS uv

# --- build: resolve exactly the locked dependencies into /app/.venv ----------------------
FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY gate ./gate
RUN uv sync --frozen --no-dev

# --- runtime: virtualenv and gate/ only; no compilers, no curl, no tests -----------------
FROM ${PYTHON_IMAGE} AS runtime
RUN groupadd --gid 10001 gateway && useradd --uid 10001 --gid 10001 --no-create-home gateway \
    && mkdir -p /data && chown 10001:10001 /data
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/gate /app/gate
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_GATEWAY_CONFIG=/etc/api-gateway/config.yaml \
    API_GATEWAY_DB=/data/api-gateway.db \
    API_GATEWAY_HOST=0.0.0.0
USER 10001:10001
EXPOSE 8787
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('API_GATEWAY_PORT', '8787') + '/healthz', timeout=2)"]
ENTRYPOINT ["api-gateway"]
CMD ["serve"]

# --- bench: runtime plus the mock upstream and load scripts ------------------------------
FROM runtime AS bench
COPY bench /app/bench

# --- default target: the runtime image (`docker build .` and `docker compose build`) ----
FROM runtime
