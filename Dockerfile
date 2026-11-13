FROM python:3.13.5-slim-bookworm AS builder
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
RUN pip install --no-cache-dir uv==0.9.5
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable --no-cache

FROM python:3.13.5-slim-bookworm
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp \
    TMPDIR=/tmp XDG_CACHE_HOME=/tmp/.cache MPLCONFIGDIR=/tmp/.mpl \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COPY --from=builder /app/.venv /app/.venv
COPY run_agent.py ./
ENTRYPOINT ["python", "/app/run_agent.py"]
