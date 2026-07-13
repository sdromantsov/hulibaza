FROM python:3.13-slim

WORKDIR /app

# Install the package + runtime deps from wheels (psycopg[binary], pymupdf,
# qdrant-client, tokenizers, mcp all ship manylinux wheels).
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# tokenizer.json files referenced by config.yaml (models[*].tokenizer_path).
# Bake them in or mount at runtime to /app/tokenizers.
#   COPY tokenizers/ /app/tokenizers/

ENV CONFIG_PATH=/app/config.yaml
EXPOSE 8080

# Streamable-HTTP MCP server on 0.0.0.0:8080.
CMD ["python", "-m", "hulibaza.server"]
