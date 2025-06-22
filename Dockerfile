FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
COPY . .

RUN uv venv && uv sync

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["kwak"]
CMD ["--help"]
