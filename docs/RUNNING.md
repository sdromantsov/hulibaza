# Running hulibaza

## Local dev / tests

```bash
pip install -e ".[dev]"

docker compose up -d postgres qdrant     # data stores (PG:59942, Qdrant:59943)

pytest                     # full suite (uses the compose Postgres)
pytest -m "not integration"  # no infra needed (unit only)
```

Integration tests auto-skip if Postgres isn't reachable. Override its URL with
`HULIBAZA_TEST_PG`.

## Running the server

The stack has its own ports/volumes/creds (never collides with the prototype):
Postgres `59942`, Qdrant `59943`, embedder `59944`, server `59980`.

1. **Data stores** — `docker compose up -d postgres qdrant`.
2. **Config** — `cp config.example.yaml config.yaml`, edit URLs / `models` / `defaults`.
3. **Tokenizers** — put each model's `tokenizer.json` in `./tokenizers/` (paths
   must match `models[*].tokenizer_path`). Token counts are computed locally.
4. **Wiki** — put sections under `wiki_dir` (default `/data/docs`, mounted
   from the host); each section is a subdirectory with a `section.yaml`.
   Optional per-section `.hulibazaignore` / `.hulibazaallow` extend the shipped
   defaults.
5. **Embedder + reranker** — start the GPU router (`docker compose up -d
   llama-server`) with models in `./llama_server/`; it serves both
   `/v1/embeddings` and `/v1/rerank` (the reranker section in `models.ini` uses
   `reranking = true`). Or point `embedding_url` at any OpenAI-compatible
   server that implements both endpoints.

   > **Sizing.** llama.cpp pre-allocates a KV cache for `ctx-size` tokens and
   > activation buffers for `ubatch-size`, both up front. Pooling models
   > (embedding/rerank) process one whole input per forward pass, so both must
   > be >= your longest input — this repo pins `parallel = 1` and
   > `ctx-size = ubatch-size = 1024` in `models.ini`. A too-small `ubatch-size`
   > rejects long inputs (*"input (N tokens) is too large to process"*); a
   > too-small `ctx-size` fails requests (*"Context size has been exceeded"*).
   > `batch-size` is logical (near-free); keep it >= `ubatch-size`.
6. **Server** — `docker compose up -d hulibaza` (streamable-HTTP MCP on
   `59980`). It builds the manager and starts the lifecycle daemon; ingestion
   is never auto-started.

To run the server on the host instead of in compose, set the URLs in
`config.yaml` to `localhost` with the mapped ports and run
`CONFIG_PATH=config.yaml python -m hulibaza.server`.

## Ingest / search

Ingestion and retrieval are MCP tools (`ingest`, `search`, `sections`,
`section_details`, `list_files`, `get_chunks`, `status`) — call them from an MCP
client. `ingest("all")` runs in the background; poll `status()`.

## Purge (destructive, human-only)

```bash
docker compose exec hulibaza python -m hulibaza.purge
```

Deletes tombstones past their grace (`in_use=false AND delete_after ≤ now`) from
Qdrant + Postgres. Skips any section mid-ingest. Never automatic.
