FROM python:3.11-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY training_service ./training_service
COPY scripts ./scripts
CMD ["uv", "run", "python", "-m", "training_service.train"]

